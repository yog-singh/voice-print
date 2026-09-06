import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .config import Config, EncoderConfig, PreprocessConfig
from .exceptions import VoicefingerprintError
from .logging_utils import set_level
from .models import DEFAULT_MODEL, REGISTRY
from .recognizer import VoiceRecognizer
from .store import SpeakerBook


def _build(args) -> VoiceRecognizer:
    config = Config(
        encoder=EncoderConfig(model=args.model, weights=args.weights),
        preprocess=PreprocessConfig(vad_backend=args.vad),
    )
    book = None
    if getattr(args, "book", None) and Path(args.book).with_suffix(".npz").is_file():
        book = SpeakerBook.load(args.book, expect_model=args.model)
    return VoiceRecognizer(config, book=book, threshold=getattr(args, "threshold", None))


def _emit(payload) -> None:
    print(json.dumps(payload, indent=2, default=float))


def cmd_models(args):
    _emit({name: {"repo": s.repo_id, "dim": s.embedding_dim, "notes": s.notes} for name, s in REGISTRY.items()})


def cmd_selfcheck(args):
    recognizer = _build(args)
    info = recognizer.encoder.describe()
    tone = 0.1 * np.sin(2 * np.pi * 220 * np.arange(int(3 * recognizer.encoder.sample_rate)) / recognizer.encoder.sample_rate)
    embedding = recognizer.encoder.embed_utterance(tone.astype(np.float32))
    info["smoke_test"] = {"shape": list(embedding.shape), "norm": round(float(np.linalg.norm(embedding)), 6)}
    _emit(info)


def cmd_embed(args):
    recognizer = _build(args)
    embedding = recognizer.embed(args.audio)
    if args.out:
        np.save(args.out, embedding)
        _emit({"saved": args.out, "dim": int(embedding.shape[0])})
    else:
        _emit({"dim": int(embedding.shape[0]), "embedding": embedding.round(6).tolist()})


def cmd_enroll(args):
    recognizer = _build(args)
    speaker = recognizer.enroll(args.name, args.audio)
    path = recognizer.save(args.book)
    _emit({"speaker": speaker.name, "utterances": speaker.n_utterances,
           "self_consistency": speaker.self_consistency(), "book": str(path)})


def cmd_list(args):
    book = SpeakerBook.load(args.book)
    _emit({"model": book.model, "speakers": book.summary()})


def cmd_verify(args):
    recognizer = _build(args)
    result = recognizer.verify(args.a, args.b)
    _emit({"score": result.score, "threshold": result.threshold,
           "accepted": result.accepted, "score_normalized": result.normalized})
    return 0 if result.accepted else 1


def cmd_identify(args):
    recognizer = _build(args)
    matches = recognizer.identify(args.audio, top_k=args.top_k)
    _emit({"threshold": recognizer.threshold,
           "matches": [{"name": m.name, "score": m.score, "accepted": m.accepted} for m in matches]})


def cmd_calibrate(args):
    recognizer = _build(args)
    _emit(recognizer.calibrate())


def cmd_diarize(args):
    from .diarize import assign_to_enrolled, diarize

    recognizer = _build(args)
    if args.enrolled:
        turns = assign_to_enrolled(recognizer, args.audio, rate=args.rate)
    else:
        turns = diarize(recognizer, args.audio, rate=args.rate, n_speakers=args.n_speakers)
    _emit([{"speaker": t.speaker, "start": round(t.start, 2), "end": round(t.end, 2)} for t in turns])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voicefingerprint", description="Speaker recognition from the command line")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(REGISTRY))
    parser.add_argument("--weights", help="path to a local .onnx, overriding the registry")
    parser.add_argument("--vad", default="energy", choices=["energy", "silero", "none"])
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, fn, help_text, book=False, threshold=False):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=fn)
        if book:
            p.add_argument("--book", default="speakers", help="speaker book path (.npz)")
        if threshold:
            p.add_argument("--threshold", type=float, help="override the decision threshold")
        return p

    add("models", cmd_models, "list available encoders")
    add("selfcheck", cmd_selfcheck, "load the model and run a smoke test")

    p = add("embed", cmd_embed, "print or save an embedding")
    p.add_argument("audio")
    p.add_argument("--out", help="write the embedding to a .npy file")

    p = add("enroll", cmd_enroll, "add a speaker to the book", book=True)
    p.add_argument("name")
    p.add_argument("audio", nargs="+")

    p = add("list", cmd_list, "show enrolled speakers", book=True)

    p = add("verify", cmd_verify, "compare two clips", threshold=True)
    p.add_argument("a")
    p.add_argument("b")

    p = add("identify", cmd_identify, "match a clip against the book", book=True, threshold=True)
    p.add_argument("audio")
    p.add_argument("--top-k", type=int, default=3)

    add("calibrate", cmd_calibrate, "fit a threshold from the enrolled speakers", book=True)

    p = add("diarize", cmd_diarize, "segment a recording by speaker", book=True, threshold=True)
    p.add_argument("audio")
    p.add_argument("--rate", type=float, default=4.0, help="embeddings per second")
    p.add_argument("--n-speakers", type=int, help="fix the speaker count instead of using a threshold")
    p.add_argument("--enrolled", action="store_true", help="label turns with enrolled names")

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.verbose:
        set_level("DEBUG")
    try:
        return args.func(args) or 0
    except VoicefingerprintError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
