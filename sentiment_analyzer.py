"""Chinese emotion analysis using Johnson8187/Chinese-Emotion-Small.

Provides local model-based emotion detection as a complement to the
rule-based EmotionDetector. Lazy-loads the model on first use and
caches it in memory for subsequent calls.

Emotion labels (8 classes):
    0: 平淡语气  (neutral)
    1: 关切语调  (caring)
    2: 开心语调  (happy)
    3: 愤怒语调  (angry)
    4: 悲伤语调  (sad)
    5: 疑问语调  (questioning)
    6: 惊奇语调  (surprised)
    7: 厌恶语调  (disgusted)
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

MODEL_ID = "Johnson8187/Chinese-Emotion-Small"
MODEL_CACHE_DIR = None  # will be set to ~/.hermes/models at runtime

# Label index -> (label_name, valence)
# valence: positive=1, neutral=0, negative=-1
LABEL_MAP: Dict[int, Tuple[str, int]] = {
    0: ("neutral",     0),
    1: ("caring",      1),
    2: ("happy",       1),
    3: ("angry",      -1),
    4: ("sad",        -1),
    5: ("questioning", 0),
    6: ("surprised",   1),
    7: ("disgusted",  -1),
}

# Mapping from model label -> emotion trigger type (for rule-miss补救)
LABEL_TO_TRIGGER: Dict[str, Optional[str]] = {
    "neutral":     None,
    "caring":      "care",
    "happy":       "praise",       # happy without rule match → treat as mild praise
    "angry":       "criticism",
    "sad":         "care",         # sad user → trigger care response
    "questioning": None,
    "surprised":   None,
    "disgusted":   "criticism",
}

# Scale factors applied to rule-detected deltas based on model emotion
# (rule_trigger, model_label) -> scale
FUSION_SCALE: Dict[Tuple[str, str], float] = {
    # Intimacy + positive emotion → amplify
    ("intimacy",   "happy"):      1.4,
    ("intimacy",   "caring"):     1.2,
    ("intimacy",   "surprised"):  1.1,
    ("intimacy",   "neutral"):    0.85,
    ("intimacy",   "questioning"):0.8,
    ("intimacy",   "sad"):        1.1,   # sad + intimacy = vulnerable, still meaningful
    ("intimacy",   "angry"):      0.5,
    ("intimacy",   "disgusted"):  0.3,

    # Praise + positive → amplify
    ("praise",     "happy"):      1.3,
    ("praise",     "caring"):     1.1,
    ("praise",     "neutral"):    0.9,
    ("praise",     "questioning"):0.8,
    ("praise",     "angry"):      0.6,
    ("praise",     "disgusted"):  0.4,

    # Care + caring/sad → amplify
    ("care",       "caring"):     1.3,
    ("care",       "sad"):        1.2,
    ("care",       "happy"):      1.0,
    ("care",       "neutral"):    0.9,
    ("care",       "angry"):      0.7,

    # Criticism + angry/disgusted → amplify (more negative impact)
    ("criticism",  "angry"):      1.5,
    ("criticism",  "disgusted"):  1.4,
    ("criticism",  "neutral"):    0.9,
    ("criticism",  "happy"):      0.6,   # criticism but happy tone → maybe joking

    # Teasing + happy → amplify
    ("teasing",    "happy"):      1.3,
    ("teasing",    "surprised"):  1.1,
    ("teasing",    "neutral"):    0.9,
    ("teasing",    "angry"):      0.5,

    # Other_ai + angry/disgusted → amplify possessiveness spike
    ("other_ai_mentioned", "angry"):    1.4,
    ("other_ai_mentioned", "disgusted"):1.3,
    ("other_ai_mentioned", "happy"):    1.6,  # happy about other AI = worse
    ("other_ai_mentioned", "neutral"):  1.0,

    # Ignored + angry → amplify
    ("ignored",    "angry"):      1.4,
    ("ignored",    "disgusted"):  1.3,
    ("ignored",    "neutral"):    1.0,
    ("ignored",    "happy"):      0.7,
}

DEFAULT_SCALE = 1.0  # fallback when no specific rule defined


@dataclass
class SentimentResult:
    """Result from the sentiment model."""
    label: str           # e.g. "happy", "angry"
    label_zh: str        # e.g. "开心语调"
    confidence: float    # 0.0-1.0 softmax probability of top class
    valence: int         # +1, 0, -1
    all_scores: Dict[str, float]  # label -> probability for all 8 classes
    inference_ms: float  # inference time in milliseconds


class SentimentAnalyzer:
    """Lazy-loading wrapper around Chinese-Emotion-Small.

    Thread-safety: not guaranteed. Designed for single-threaded
    sequential use in the emotion state update pipeline.
    """

    _instance: Optional["SentimentAnalyzer"] = None

    def __init__(self, model_cache_dir: Optional[str] = None):
        self._model = None
        self._tokenizer = None
        self._model_cache_dir = model_cache_dir
        self._load_attempted = False
        self._available = False

    @classmethod
    def get_instance(cls, model_cache_dir: Optional[str] = None) -> "SentimentAnalyzer":
        """Return singleton instance."""
        if cls._instance is None:
            cls._instance = cls(model_cache_dir=model_cache_dir)
        return cls._instance

    def _try_load(self) -> bool:
        """Attempt to load model. Returns True if successful."""
        if self._load_attempted:
            return self._available

        self._load_attempted = True
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            logger.info("Loading Chinese-Emotion-Small model...")
            t0 = time.time()

            kwargs = {}
            if self._model_cache_dir:
                kwargs["cache_dir"] = self._model_cache_dir

            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **kwargs)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_ID, **kwargs
            )
            self._model.eval()
            self._torch = torch

            elapsed = time.time() - t0
            logger.info(f"Chinese-Emotion-Small loaded in {elapsed:.1f}s")
            self._available = True
            return True

        except Exception as e:
            logger.warning(f"Chinese-Emotion-Small unavailable: {e}. "
                           "Emotion system will use rules only.")
            self._available = False
            return False

    @property
    def available(self) -> bool:
        """True if model is loaded and ready."""
        if not self._load_attempted:
            self._try_load()
        return self._available

    def analyze(self, text: str) -> Optional[SentimentResult]:
        """Run emotion classification on text.

        Returns None if model is unavailable or text is empty.
        """
        if not text or not text.strip():
            return None

        if not self._try_load():
            return None

        try:
            t0 = time.time()
            torch = self._torch

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            )

            with torch.no_grad():
                outputs = self._model(**inputs)

            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item())

            label_name, valence = LABEL_MAP[pred_idx]
            label_zh_map = {
                "neutral": "平淡语气", "caring": "关切语调", "happy": "开心语调",
                "angry": "愤怒语调", "sad": "悲伤语调", "questioning": "疑问语调",
                "surprised": "惊奇语调", "disgusted": "厌恶语调",
            }

            all_scores = {
                LABEL_MAP[i][0]: float(probs[i].item())
                for i in range(len(LABEL_MAP))
            }

            elapsed_ms = (time.time() - t0) * 1000

            return SentimentResult(
                label=label_name,
                label_zh=label_zh_map.get(label_name, label_name),
                confidence=confidence,
                valence=valence,
                all_scores=all_scores,
                inference_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return None

    def get_fusion_scale(self, trigger_type: str, sentiment: Optional[SentimentResult]) -> float:
        """Get delta scale factor based on rule trigger + model sentiment.

        Args:
            trigger_type: Rule-detected trigger (e.g. "intimacy", "praise")
            sentiment: Model output, or None if unavailable

        Returns:
            Scale factor to multiply emotion deltas by (0.3 - 1.6 range)
        """
        if sentiment is None:
            return DEFAULT_SCALE

        key = (trigger_type, sentiment.label)
        base_scale = FUSION_SCALE.get(key, DEFAULT_SCALE)

        # Weight by model confidence: low confidence → pull toward 1.0
        # scale = 1.0 + (base_scale - 1.0) * confidence
        confidence_weight = sentiment.confidence
        weighted_scale = 1.0 + (base_scale - 1.0) * confidence_weight

        return round(weighted_scale, 3)

    def get_fallback_trigger(self, sentiment: Optional[SentimentResult]) -> Optional[str]:
        """Get a fallback trigger type when rules found nothing but model detected emotion.

        Only fires when model confidence is high enough (>= 0.6).

        Returns:
            Trigger type string, or None
        """
        if sentiment is None:
            return None
        if sentiment.confidence < 0.6:
            return None
        if sentiment.label in ("neutral", "questioning"):
            return None
        return LABEL_TO_TRIGGER.get(sentiment.label)
