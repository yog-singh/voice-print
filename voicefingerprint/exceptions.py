class VoicefingerprintError(Exception):
    """Base class for all library errors."""


class AudioError(VoicefingerprintError):
    """Audio could not be loaded, decoded or resampled."""


class ModelError(VoicefingerprintError):
    """Model weights are missing, unreadable or have an unexpected signature."""


class FeatureError(VoicefingerprintError):
    """Audio was too short or otherwise unusable for feature extraction."""


class EnrollmentError(VoicefingerprintError):
    """Speaker enrollment or lookup failed."""
