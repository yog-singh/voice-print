from .audio import load_audio, normalize_volume, preprocess, resample
from .chunking import compute_chunks
from .config import ChunkConfig, Config, EncoderConfig, FbankConfig, PreprocessConfig
from .diarize import Turn, assign_to_enrolled, diarize
from .encoder import SpeakerEncoder
from .exceptions import (
    AudioError,
    EnrollmentError,
    FeatureError,
    ModelError,
    VoiceprintError,
)
from .features import fbank
from .logging_utils import set_level
from .models import REGISTRY, get_spec
from .recognizer import Match, VerificationResult, VoiceRecognizer
from .scoring import ASNorm, calibrate_threshold, cosine, cosine_matrix, equal_error_rate
from .store import Speaker, SpeakerBook

__version__ = "0.1.0"

__all__ = [
    "ASNorm",
    "AudioError",
    "ChunkConfig",
    "Config",
    "EncoderConfig",
    "EnrollmentError",
    "FbankConfig",
    "FeatureError",
    "Match",
    "ModelError",
    "PreprocessConfig",
    "REGISTRY",
    "Speaker",
    "SpeakerBook",
    "SpeakerEncoder",
    "Turn",
    "VerificationResult",
    "VoiceRecognizer",
    "VoiceprintError",
    "assign_to_enrolled",
    "calibrate_threshold",
    "compute_chunks",
    "cosine",
    "cosine_matrix",
    "diarize",
    "equal_error_rate",
    "fbank",
    "get_spec",
    "load_audio",
    "normalize_volume",
    "preprocess",
    "resample",
    "set_level",
]
