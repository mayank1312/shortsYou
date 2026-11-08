import logging

import numpy as np
import shap
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

from app.features.train_cpep.db import FEATURE_ORDER, cpep_models_db
from app.features.train_cpep.schema import TrainCpepRequest

logger = logging.getLogger(__name__)

GLOBAL_MODEL_USER_ID = "global-model"
MIN_SAMPLES_TO_TRAIN = 2
MIN_SAMPLES_GLOBAL = 30  # pooled model needs many more clips
MIN_SAMPLES_FOR_HELD_OUT_TEST = 5
TEST_SPLIT_FRACTION = 0.2


def _pearson_r(y_true, y_pred) -> float:
    if len(y_true) < 2:
        return 0.0
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _send(payload: dict) -> None:
    """No callback for train-cpep: the result is saved to Mongo and logged."""
    if payload.get("error"):
        logger.error("CPEP training result | %s", payload)
    else:
        logger.info("CPEP training result | %s", payload)


def train_cpep_model(request: TrainCpepRequest) -> None:
    """Background task. Every outcome (success or failure) is logged; no callback."""
    try:
        n = len(request.training_data)
        is_global = request.user_id == GLOBAL_MODEL_USER_ID
        minimum = MIN_SAMPLES_GLOBAL if is_global else MIN_SAMPLES_TO_TRAIN

        if n < minimum:
            kind = "global model" if is_global else "model"
            _send({
                "job_id": request.job_id,
                "user_id": request.user_id,
                "error": f"Not enough training clips for {kind} (got {n}, need at least {minimum})",
            })
            return

        features = np.array(
            [
                [c.emotion_score, c.semantic_score, c.hook_score, c.duration_seconds]
                for c in request.training_data
            ],
            dtype=float,
        )
        targets = np.log1p(
            np.array([max(c.views_48h, 0.0) for c in request.training_data], dtype=float)
        )

        if n >= MIN_SAMPLES_FOR_HELD_OUT_TEST:
            n_test = max(1, round(n * TEST_SPLIT_FRACTION))
            split = n - n_test
            train_x, test_x = features[:split], features[split:]
            train_y, test_y = targets[:split], targets[split:]
        else:
            train_x, test_x = features, features
            train_y, test_y = targets, targets

        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
        model.fit(train_x, train_y)

        preds = model.predict(test_x)
        rmse = float(np.sqrt(mean_squared_error(test_y, preds)))
        r2 = float(r2_score(test_y, preds)) if len(test_y) >= 2 else 0.0
        pearson = _pearson_r(test_y, preds)

        importances = []
        try:
            sv = shap.TreeExplainer(model).shap_values(train_x)
            mean_abs = np.abs(sv).mean(axis=0)
            importances = [
                {"feature": f, "importance": float(v)}
                for f, v in zip(FEATURE_ORDER, mean_abs)
            ]
            importances.sort(key=lambda x: x["importance"], reverse=True)
        except Exception:
            logger.exception("SHAP failed - continuing without importances")

        model_json = model.get_booster().save_raw(raw_format="json").decode("utf-8")
        version = cpep_models_db.get_latest_version_number(request.user_id) + 1

        cpep_models_db.save_model(
            job_id=request.job_id,
            user_id=request.user_id,
            version_number=version,
            model_json=model_json,
            pearson_r=pearson,
            rmse=rmse,
            r2=r2,
            feature_importances=importances,
            n_samples=n,
        )

        _send({
            "job_id": request.job_id,
            "user_id": request.user_id,
            "model_version": f"v{version}",
            "pearson_r": pearson,
            "rmse": rmse,
            "r2": r2,
            "feature_importances": importances,
            "n_samples": n,
            "error": "",
        })
    except Exception as exc:
        logger.exception("CPEP training failed | job_id=%s", request.job_id)
        _send({
            "job_id": request.job_id,
            "user_id": request.user_id,
            "error": str(exc),
        })