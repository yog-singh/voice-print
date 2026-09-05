class VoiceprintError(Exception):
    """Base class for all library errors."""


class AudioError(VoiceprintError):
    """Audio could not be loaded, decoded or resampled."""


class ModelError(VoiceprintError):
    """Model weights are missing, unreadable or have an unexpected signature."""


class FeatureError(VoiceprintError):
    """Audio was too short or otherwise unusable for feature extraction."""


class EnrollmentError(VoiceprintError):
    """Speaker enrollment or lookup failed."""
