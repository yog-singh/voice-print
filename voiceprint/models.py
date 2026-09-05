import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .config import FbankConfig
from .exceptions import ModelError
from .logging_utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    filename: str
    embedding_dim: int
    input_name: str
    output_name: str
    fbank: FbankConfig
    scale_to_int16: bool  # WeSpeaker computes fbank on int16-scaled samples
    default_threshold: float
    notes: str = ""


REGISTRY: Dict[str, ModelSpec] = {
    "wespeaker_resnet34_lm": ModelSpec(
        name="wespeaker_resnet34_lm",
        repo_id="Wespeaker/wespeaker-voxceleb-resnet34-LM",
        filename="voxceleb_resnet34_LM.onnx",
        embedding_dim=256,
        input_name="feats",
        output_name="embs",
        fbank=FbankConfig(),
        scale_to_int16=True,
        default_threshold=0.40,
        notes="ResNet34 trained on VoxCeleb2 with large-margin finetuning. ~0.7% EER on Vox1-O.",
    ),
}

DEFAULT_MODEL = "wespeaker_resnet34_lm"


def get_spec(name: str) -> ModelSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise ModelError(
            f"Unknown model {name!r}. Available: {sorted(REGISTRY)}"
        ) from None


def resolve_weights(spec: ModelSpec, override: Optional[str] = None) -> Path:
    """Return a local path to the ONNX file, downloading it on first use."""
    candidate = override or os.environ.get("VOICEPRINT_WEIGHTS")
    if candidate:
        path = Path(candidate).expanduser()
        if not path.is_file():
            raise ModelError(f"Weights file not found: {path}")
        return path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ModelError(
            "huggingface_hub is required to fetch weights automatically. "
            "Install it, or pass an explicit path via EncoderConfig(weights=...)."
        ) from None

    log.info("resolving %s/%s from the Hugging Face cache", spec.repo_id, spec.filename)
    try:
        return Path(hf_hub_download(spec.repo_id, spec.filename))
    except Exception as exc:
        raise ModelError(f"Could not fetch {spec.repo_id}/{spec.filename}: {exc}") from exc
