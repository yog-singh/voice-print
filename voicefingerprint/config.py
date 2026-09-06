from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class FbankConfig:
    """Kaldi-compatible filterbank settings. Must match what the encoder was trained on."""

    sample_rate: int = 16000
    n_mels: int = 80
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    low_freq: float = 20.0
    high_freq: float = 0.0  # <=0 means an offset from the Nyquist frequency
    preemphasis: float = 0.97
    dither: float = 0.0
    remove_dc_offset: bool = True
    window: str = "hamming"
    snip_edges: bool = True
    cmn: bool = True

    @property
    def frame_length(self) -> int:
        return int(self.sample_rate * self.frame_length_ms / 1000)

    @property
    def frame_shift(self) -> int:
        return int(self.sample_rate * self.frame_shift_ms / 1000)

    @property
    def padded_length(self) -> int:
        n = 1
        while n < self.frame_length:
            n *= 2
        return n


@dataclass(frozen=True)
class ChunkConfig:
    """How long audio is split into fixed windows before embedding."""

    window_s: float = 3.0
    rate: float = 0.75  # windows emitted per second; 1/window_s means no overlap
    min_coverage: float = 0.6

    def window_samples(self, sample_rate: int) -> int:
        return int(round(self.window_s * sample_rate))

    def step_samples(self, sample_rate: int) -> int:
        step = int(round(sample_rate / self.rate))
        return max(1, min(step, self.window_samples(sample_rate)))


@dataclass(frozen=True)
class PreprocessConfig:
    sample_rate: int = 16000
    target_dbfs: float = -26.0
    normalize_volume: bool = True
    trim_silence: bool = True
    vad_backend: str = "energy"  # "energy" | "silero" | "none"


@dataclass
class EncoderConfig:
    model: str = "wespeaker_resnet34_lm"
    weights: Optional[str] = None  # overrides the registry lookup
    providers: Optional[List[str]] = None
    batch_size: int = 16
    intra_op_threads: int = 0  # 0 lets onnxruntime decide


@dataclass
class Config:
    """Top-level configuration handed to VoiceRecognizer."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
