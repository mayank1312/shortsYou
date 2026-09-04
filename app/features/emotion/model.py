import logging

import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)


logger = logging.getLogger(__name__)


class EmotionModel:
    

    MODEL_NAME = "superb/wav2vec2-base-superb-er"

    SAMPLE_RATE = 16000

    def __init__(self) -> None:

        logger.info(
            "Loading emotion classification model | model=%s",
            self.MODEL_NAME,
        )

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.MODEL_NAME
        )

        self.model = AutoModelForAudioClassification.from_pretrained(
            self.MODEL_NAME
        )

        self.model.eval()

        self.id2label = self.model.config.id2label

    def classify(
        self,
        waveform,
    ) -> list[dict]:
        """
        Runs inference on an already-loaded 16kHz mono waveform
        (numpy float32 array). Returns [{"label": ..., "score": ...}]
        for every class, unsorted - same shape as pipeline() output
        so calling code doesn't need to change.
        """

        inputs = self.feature_extractor(
            waveform,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits

        scores = torch.softmax(logits, dim=-1)[0]

        return [
            {
                "label": self.id2label[index],
                "score": float(scores[index]),
            }
            for index in range(len(scores))
        ]


emotion_model = EmotionModel()