import logging

import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)


logger = logging.getLogger(__name__)


class EmotionModel:
    """
    Abbas's original spec named facebook/wav2vec2-base for this, but
    that checkpoint has no emotion classification head - it is the
    raw pretrained encoder used for ASR pretraining. Running it
    through pipeline("audio-classification", ...) would attach a
    randomly-initialized classification head and produce meaningless,
    effectively random labels.

    superb/wav2vec2-base-superb-er is an actual emotion-fine-tuned
    checkpoint (IEMOCAP dataset: neutral, happy, angry, sad) and is
    what makes this pipeline produce real output.

    This loads the feature extractor + model directly instead of
    using transformers.pipeline(). Newer transformers versions route
    pipeline()'s audio-classification through torchcodec for
    decoding, which needs matching native FFmpeg shared libraries
    that aren't guaranteed to exist in a minimal deploy environment
    (and don't need to - we already have a raw waveform array from
    librosa by the time we get here). Calling the model directly
    avoids that dependency entirely.
    """

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

