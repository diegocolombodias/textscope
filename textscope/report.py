"""Assemble stylometric and (optional) surprisal features into a report."""

from __future__ import annotations

from typing import Optional

from .stylometry import analyze_style, split_sentences, StyleReport

DISCLAIMER = """\
WHAT THIS REPORT IS NOT
  This tool does not determine authorship and does not output a
  probability that text was machine-written. No such number can be
  computed honestly from these features.

  Formal academic prose and prose written in a second language both
  score as low-surprisal for reasons unrelated to authorship. Published
  research found detectors misclassifying the majority of TOEFL essays
  by non-native speakers while scoring near-perfectly on native ones.
  Any tool of this kind, including this one, inherits that failure mode.

  Use the output to locate passages worth reading closely. Then read
  them, and if the text is someone else's, talk to them.
"""


def build_report(
    text: str,
    lang: str = "en",
    scorer=None,
    per_sentence_surprisal: bool = True,
) -> dict:
    """
    scorer: an optional LocalScorer instance. Omit it to run the
    model-free stylometric analysis only (no torch required).
    """
    style: StyleReport = analyze_style(text, lang=lang)
    out: dict = {
        "style": style.as_dict(),
        "sentences": [
            {
                "index": f.index,
                "n_words": f.n_words,
                "flags": f.flags,
                "text": f.text,
            }
            for f in style.sentences
        ],
    }

    if scorer is not None:
        doc = scorer.score(text)
        out["surprisal"] = doc.as_dict()

        if per_sentence_surprisal:
            sents = split_sentences(text)
            reports = scorer.score_sentences(sents)
            for entry, rep in zip(out["sentences"], reports):
                if rep is not None:
                    entry["mean_nll"] = round(rep.mean_nll, 3)
                    entry["perplexity"] = round(rep.perplexity, 2)

            valid = [e for e in out["sentences"] if "mean_nll" in e]
            if valid:
                ranked = sorted(valid, key=lambda e: e["mean_nll"])
                out["lowest_surprisal_sentences"] = [
                    e["index"] for e in ranked[:3]
                ]

    return out


def render_text(report: dict, show_disclaimer: bool = True) -> str:
    lines: list[str] = []
    add = lines.append

    add("=" * 72)
    add("TEXTSCOPE REPORT")
    add("=" * 72)
    add("")

    add("DOCUMENT-LEVEL FEATURES")
    for k, v in report["style"].items():
        if isinstance(v, dict):
            if v:
                items = ", ".join(f"{kk} ({vv})" for kk, vv in v.items())
                add(f"  {k:<32} {items}")
        else:
            add(f"  {k:<32} {v}")

    if "surprisal" in report:
        add("")
        add("SURPRISAL FEATURES (relative to the local scoring model)")
        for k, v in report["surprisal"].items():
            add(f"  {k:<32} {v}")

    add("")
    add("PER-SENTENCE NOTES")
    any_flag = False
    for e in report["sentences"]:
        bits = []
        if e["flags"]:
            bits.append("; ".join(e["flags"]))
        if "perplexity" in e:
            bits.append(f"ppl={e['perplexity']}")
        if not bits:
            continue
        any_flag = True
        add(f"  [s{e['index']:>2}] {e['n_words']:>3}w  {' | '.join(bits)}")
        add(f"         {e['text'][:88]}{'...' if len(e['text']) > 88 else ''}")
    if not any_flag:
        add("  (nothing flagged)")

    if "lowest_surprisal_sentences" in report:
        add("")
        idx = ", ".join(f"s{i}" for i in report["lowest_surprisal_sentences"])
        add(f"MOST PREDICTABLE SENTENCES: {idx}")

    if show_disclaimer:
        add("")
        add("-" * 72)
        add(DISCLAIMER)

    return "\n".join(lines)
