#!/usr/bin/env python3
"""
TextScope CLI.

Examples
--------
Stylometry only (no model, no torch needed):
    python cli.py analyze paper.txt --lang en

With a local language model for surprisal:
    python cli.py analyze paper.txt --model ./models/gpt2-large

Build a reference corpus from your own past writing:
    python cli.py calibrate ./my_published_papers/*.txt \\
        --model ./models/gpt2-large --out reference.json

Compare a new draft against that reference:
    python cli.py analyze draft.txt --model ./models/gpt2-large \\
        --reference reference.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from textscope.report import build_report, render_text          # noqa: E402
from textscope.stylometry import analyze_style                  # noqa: E402
from textscope import calibration                               # noqa: E402


def _load_scorer(model_path: str | None, device: str):
    if not model_path:
        return None
    try:
        from textscope.perplexity import LocalScorer
    except ImportError as exc:
        sys.exit(
            f"Could not import the scoring backend ({exc}).\n"
            "Install it with:  pip install torch transformers\n"
            "Or omit --model to run the stylometric analysis only."
        )
    print(f"Loading model from {model_path} ... ", end="", flush=True)
    scorer = LocalScorer(model_path=model_path, device=device)
    print("done.")
    return scorer


def _doc_features(text: str, lang: str, scorer) -> dict[str, float]:
    feats: dict[str, float] = {}
    style = analyze_style(text, lang=lang).as_dict()
    for k, v in style.items():
        if isinstance(v, (int, float)):
            feats[k] = float(v)
    if scorer is not None:
        for k, v in scorer.score(text).as_dict().items():
            feats[k] = float(v)
    return feats


def cmd_analyze(args) -> None:
    text = Path(args.path).read_text(encoding="utf-8")
    scorer = _load_scorer(args.model, args.device)

    report = build_report(text, lang=args.lang, scorer=scorer)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print(render_text(report, show_disclaimer=not args.no_disclaimer))

    if args.reference:
        ref = calibration.ReferenceStats.load(args.reference)
        feats = _doc_features(text, args.lang, scorer)
        z = calibration.z_scores(feats, ref)
        print()
        print("=" * 72)
        print(f"CALIBRATION vs {args.reference} "
              f"({ref.n_documents} reference documents)")
        print("=" * 72)
        for line in calibration.interpret(z):
            print(line)


def cmd_calibrate(args) -> None:
    paths: list[str] = []
    for pattern in args.paths:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        sys.exit("No files matched.")

    scorer = _load_scorer(args.model, args.device)

    rows = []
    for i, p in enumerate(paths, 1):
        text = Path(p).read_text(encoding="utf-8")
        if len(text.split()) < 120:
            print(f"  skipping {p} (too short)")
            continue
        rows.append(_doc_features(text, args.lang, scorer))
        print(f"  [{i}/{len(paths)}] {p}")

    ref = calibration.build_reference(
        rows,
        model_id=args.model or "stylometry-only",
        note=args.note,
    )
    ref.save(args.out)
    print(f"\nWrote reference over {ref.n_documents} documents to {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="textscope",
        description="Local, private prose analysis. Not an authorship oracle.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyze one document")
    a.add_argument("path")
    a.add_argument("--lang", default="en", choices=["en", "pt"])
    a.add_argument("--model", default=None,
                   help="path to a local HF causal LM directory")
    a.add_argument("--device", default="cpu")
    a.add_argument("--reference", default=None,
                   help="reference.json from the calibrate command")
    a.add_argument("--json", action="store_true")
    a.add_argument("--no-disclaimer", action="store_true")
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("calibrate", help="build a reference corpus")
    c.add_argument("paths", nargs="+")
    c.add_argument("--lang", default="en", choices=["en", "pt"])
    c.add_argument("--model", default=None)
    c.add_argument("--device", default="cpu")
    c.add_argument("--out", default="reference.json")
    c.add_argument("--note", default="")
    c.set_defaults(func=cmd_calibrate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
