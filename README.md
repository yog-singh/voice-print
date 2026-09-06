# voicefingerprint

Speaker recognition, verification and diarization built on the
[WeSpeaker](https://github.com/wenet-e2e/wespeaker) ResNet34 encoder.

Waveform in, 256-d embedding out. Two clips of the same person land close together
under cosine similarity; two different people land far apart. Everything else here —
enrollment, thresholds, diarization — is built on that one property.

Inference runs on ONNX Runtime, so **PyTorch is not a dependency**. A working install is
about 50 MB plus a 27 MB model, and the encoder is CPU-friendly.

## Install

```bash
pip install -e .
```

Model weights are pulled from the Hugging Face Hub on first use and cached under
`~/.cache/huggingface`. To run fully offline, download
`Wespeaker/wespeaker-voxceleb-resnet34-LM/voxceleb_resnet34_LM.onnx` once and point at it
with `--weights` or `VOICEFINGERPRINT_WEIGHTS`.

## Quick start

```python
from voicefingerprint import VoiceRecognizer

vp = VoiceRecognizer()

vp.enroll("alice", ["alice_1.wav", "alice_2.wav", "alice_3.wav"])
vp.enroll("bob", ["bob_1.wav", "bob_2.wav"])
vp.save("speakers")

result = vp.verify("alice_1.wav", "unknown.wav")
print(result.score, result.accepted)

for match in vp.identify("unknown.wav"):
    print(match.name, round(match.score, 3), match.accepted)
```

```bash
voicefingerprint selfcheck
voicefingerprint enroll --book speakers alice alice_*.wav
voicefingerprint identify --book speakers unknown.wav
voicefingerprint verify a.wav b.wav          # exit code 0 = accepted, 1 = rejected
voicefingerprint diarize --n-speakers 3 interview.mp3
```

## How it works

```
audio file ──▶ audio.py     load, downmix, resample to 16 kHz, level to -26 dBFS
               vad.py       drop non-speech
               chunking.py  slice into overlapping 3 s windows
               features.py  80-bin Kaldi log-mel filterbank per window
               encoder.py   ResNet34 ONNX ──▶ (n_windows, 256)
                            mean-pool, L2-normalize ──▶ (256,)
```

Long audio is split into fixed windows because scores are only comparable when every
embedding summarizes a similar span. The same code path serves whole-file embedding
(`rate=0.75`) and dense diarization timelines (`rate=4`); only the window rate changes.

`features.py` reimplements `torchaudio.compliance.kaldi.fbank` in NumPy so the package
stays PyTorch-free. This is the one place where a silent bug would quietly degrade every
downstream score, so `tests/test_features.py::test_matches_torchaudio` asserts parity
against torchaudio when it is installed — measured max absolute difference is `1.6e-4`,
which is float32 rounding.

## Thresholds

`verify()` and `identify()` compare a cosine score against a threshold. The shipped
default (`0.40`) is a starting point, not an answer: the right value depends on your
microphones, languages, clip lengths and on how you weigh false accepts against false
rejects. Derive it from your own data.

```python
report = vp.calibrate()   # needs >= 2 enrolled speakers with several clips each
print(report["threshold"], report["eer"])
```

For deployments spanning several recording conditions, enable AS-norm. It rescales each
score by the statistics of both sides against a cohort of background speakers, which
makes one threshold transfer across devices far better than raw cosine does.

```python
vp.set_cohort(list(Path("background_speakers").glob("*.wav")))
```

## Measured behaviour

`scripts/evaluate.py` runs the full pipeline over a `<speaker>/<clip>` directory tree.
On the 10-speaker LibriSpeech `test-other` sample bundled with Resemblyzer, enrolling on
5 clips per speaker and testing on the other 5:

```
top-1 accuracy    100.00%   (50 test clips)
verification EER    3.01%   (1225 trials, threshold 0.343)
same-speaker mean   0.787
```

Treat the EER as a small-sample estimate. The encoder's published figure is ~0.7% EER on
the full VoxCeleb1-O benchmark.

## What this does not do

**These embeddings are not an anti-spoofing measure.** A recording of an enrolled speaker
played back into a microphone scores just as high as the live speaker, because it is
acoustically the same voice. Synthesized and converted speech can also score high. If a
decision here gates access to anything, pair it with a separate liveness or replay
detector — deliberately kept out of `verify()` so the distinction stays visible.

Diarization here is embedding clustering. It assigns one speaker per window and does not
handle overlapping speech. For overlap-aware diarization, use `pyannote.audio`.

## Layout

| Module | Responsibility |
| --- | --- |
| `audio.py` | Loading, resampling, volume normalization, the preprocessing pipeline |
| `vad.py` | `EnergyVAD` (no extra deps) and `SileroVAD` (`--vad silero`) |
| `features.py` | Kaldi-compatible log-mel filterbank |
| `chunking.py` | Window placement and coverage rules |
| `encoder.py` | ONNX session, batching, embedding pooling |
| `store.py` | `SpeakerBook` enrollment database (`.npz`) |
| `scoring.py` | Cosine, AS-norm, EER, threshold calibration |
| `recognizer.py` | `VoiceRecognizer` — the API most callers want |
| `diarize.py` | Timeline embedding, clustering, turn construction |
| `models.py` | Encoder registry and weight resolution |

Adding an encoder means adding one `ModelSpec` to `REGISTRY` in `models.py`: the repo,
the ONNX filename, its input/output tensor names, and the feature config it was trained
on. Nothing else in the package is model-specific. `encoder.py` validates the ONNX
signature against the spec at load time, so a mismatched model fails loudly instead of
emitting plausible-looking garbage.

A speaker book records which encoder produced it and refuses to load under a different
one, since embeddings from different models are not comparable.

## Development

```bash
pip install -e ".[dev]"
pytest -q
python scripts/evaluate.py path/to/speaker/dirs
VOICEFINGERPRINT_LOG=DEBUG voicefingerprint -v identify --book speakers clip.wav
```

The `dev` extra installs torch and torchaudio purely for the fbank parity test. Note that
torch 2.2 requires `numpy<2`; the parity test skips itself when the two are incompatible,
and the rest of the suite runs on either NumPy major version.

On macOS, Accelerate's BLAS leaves stale floating-point exception flags after some matmul
kernels, which makes NumPy report overflow and divide-by-zero on results that are
entirely finite. All matrix products go through `numeric.matmul`, which suppresses those
spurious warnings and raises if a product genuinely goes non-finite.
