"""Sanity-check the library on a directory of <speaker>/<utterance>.flac|wav files."""

import argparse
import sys
from pathlib import Path

import numpy as np

from voiceprint import Config, VoiceRecognizer
from voiceprint.scoring import calibrate_threshold, cosine_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="directory with one subdirectory per speaker")
    parser.add_argument("--enroll", type=int, default=5, help="utterances per speaker used for enrollment")
    parser.add_argument("--ext", default="flac")
    args = parser.parse_args()

    speakers = {d.name: sorted(d.glob(f"*.{args.ext}")) for d in sorted(Path(args.root).iterdir()) if d.is_dir()}
    speakers = {k: v for k, v in speakers.items() if len(v) > args.enroll}
    if len(speakers) < 2:
        print("need at least two speakers with enough utterances", file=sys.stderr)
        return 2

    recognizer = VoiceRecognizer(Config())
    trials, labels = [], []
    for name, files in speakers.items():
        recognizer.enroll(name, files[: args.enroll])
        for path in files[args.enroll :]:
            trials.append(recognizer.embed(path))
            labels.append(name)
    trials = np.stack(trials)

    names, centroids = recognizer.book.centroids()
    scores = cosine_matrix(trials, centroids)
    predicted = [names[i] for i in scores.argmax(axis=1)]
    accuracy = float(np.mean([p == t for p, t in zip(predicted, labels)]))

    report = calibrate_threshold(trials, labels)
    print(f"speakers          {len(speakers)}")
    print(f"enrollment clips  {args.enroll} each")
    print(f"test clips        {len(trials)}")
    print(f"top-1 accuracy    {100 * accuracy:.2f}%")
    print(f"verification EER  {100 * report['eer']:.2f}%  (threshold {report['threshold']:.3f})")
    print(f"same-speaker mean {scores.max(axis=1).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
