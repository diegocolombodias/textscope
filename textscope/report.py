"""Assemble stylometric and (optional) surprisal features into a report."""

from __future__ import annotations

from typing import Optional

from .calibration import ReferenceStats
from .stylometry import analyze_style, split_sentences, StyleReport
from .rewrite import rewrite_text

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

CLASSIFIER_DISCLAIMER = """\
ABOUT THE NUMBER ABOVE
  This is the one deliberate exception in this tool: a black-box
  probability from a supervised classifier, not an inspectable z-score.
  It was trained on the public HC3 corpus plus this project's own small,
  mixed-provenance data (see textscope/classifier.py and
  reference_corpus/) — nowhere near the scale of real, audited training
  data a commercial vendor like Turnitin uses. It has not been
  independently validated. Check eval_metrics.json next to the model
  checkpoint before trusting this number for anything, and treat it as a
  rough prior, not a verdict — everything the DISCLAIMER above says about
  false positives on formal or second-language prose applies here too,
  just hidden behind a single number instead of a number you can audit.
  English only — it was trained exclusively on English text and will
  silently return a meaningless number on anything else.
"""


def build_report(
    text: str,
    lang: str = "en",
    scorer=None,
    per_sentence_surprisal: bool = True,
    include_rewrite: bool = True,
    reference: Optional[ReferenceStats] = None,
    classifier=None,
) -> dict:
    """
    scorer: an optional LocalScorer instance. Omit it to run the
    model-free stylometric analysis only (no torch required).

    include_rewrite: also run the deterministic, rule-based rewrite
    (textscope.rewrite) and attach it under the "rewrite" key.

    reference: a ReferenceStats built with `textscope calibrate` on a
    corpus of real papers. When given, document_suggestions and the
    rewrite's "unresolved" list compare against it by z-score instead of
    the fixed heuristic thresholds — see stylometry._document_suggestions.

    classifier: an optional textscope.classifier.ClassifierScorer. When
    given, attaches an "ai_probability" key — a single black-box number,
    the deliberate exception to everything else in this module (and in
    DISCLAIMER below). See classifier.py's module docstring before using
    this for anything that matters.
    """
    style: StyleReport = analyze_style(text, lang=lang, reference=reference)
    out: dict = {
        "style": style.as_dict(),
        "document_suggestions": style.document_suggestions,
        "sentences": [
            {
                "index": f.index,
                "n_words": f.n_words,
                "flags": f.flags,
                "suggestions": f.suggestions,
                "text": f.text,
            }
            for f in style.sentences
        ],
    }

    if include_rewrite:
        out["rewrite"] = rewrite_text(text, lang=lang, reference=reference).as_dict()

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

    if classifier is not None:
        out["ai_probability"] = classifier.predict_proba(text).as_dict()

    return out


def _wrap(text: str, width: int, indent: str) -> list[str]:
    """Minimal word wrap so long suggestion sentences don't run off screen."""
    words = text.split()
    lines: list[str] = []
    cur = indent
    for w in words:
        if len(cur) + 1 + len(w) > width and cur != indent:
            lines.append(cur)
            cur = indent + w
        else:
            cur = (cur + " " + w) if cur != indent else cur + w
    if cur.strip():
        lines.append(cur)
    return lines


def render_text(
    report: dict, show_disclaimer: bool = True, show_rewrite: bool = True
) -> str:
    lines: list[str] = []
    add = lines.append

    sentences = report["sentences"]
    flagged = [e for e in sentences if e["flags"]]
    lowest = set(report.get("lowest_surprisal_sentences", []))

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
    add("SUMMARY")
    add(f"  {len(flagged)} of {len(sentences)} sentences carry at least "
        f"one stylometric flag.")

    doc_suggestions = report.get("document_suggestions", [])
    add("")
    add("DOCUMENT-LEVEL SUGGESTIONS")
    if not doc_suggestions:
        add("  (nothing triggered at the document level)")
    else:
        for i, s in enumerate(doc_suggestions, 1):
            prefix = f"  {i}. "
            wrapped = _wrap(s, 68, " " * len(prefix))
            wrapped[0] = prefix + wrapped[0].lstrip()
            lines.extend(wrapped)

    add("")
    add("PER-SENTENCE REVIEW")
    if not flagged and not report.get("lowest_surprisal_sentences"):
        add("  (nothing flagged)")
    else:
        for e in sentences:
            has_flags = bool(e["flags"])
            is_low_surprisal = e["index"] in lowest
            if not has_flags and not is_low_surprisal:
                continue

            header_bits = [f"[s{e['index']:>2}]", f"{e['n_words']}w"]
            if "perplexity" in e:
                header_bits.append(f"ppl={e['perplexity']}")
            if is_low_surprisal:
                header_bits.append("MOST PREDICTABLE")
            add("  " + "  ".join(header_bits))

            excerpt = e["text"][:100] + ("..." if len(e["text"]) > 100 else "")
            add(f"    text:  {excerpt}")

            if e["flags"]:
                add(f"    flags: {'; '.join(e['flags'])}")

            for s in e.get("suggestions", []):
                prefix = "    fix:  "
                wrapped = _wrap(s, 68, " " * len(prefix))
                wrapped[0] = prefix + wrapped[0].lstrip()
                lines.extend(wrapped)

            if is_low_surprisal and not e.get("suggestions"):
                add("    fix:  Predictable to the local model — often "
                    "generic phrasing. Reread and consider making the "
                    "claim more specific.")

            add("")

    if "lowest_surprisal_sentences" in report:
        idx = ", ".join(f"s{i}" for i in report["lowest_surprisal_sentences"])
        add(f"MOST PREDICTABLE SENTENCES: {idx}")
        add("  (flagged with MOST PREDICTABLE above; reread them for "
            "genericness — this is not proof of anything on its own)")

    if show_rewrite and "rewrite" in report:
        rw = report["rewrite"]
        add("")
        add("=" * 72)
        add("SUGGESTED REWRITE (mechanical fixes applied automatically)")
        add("=" * 72)
        add("")
        add(f"{rw['changed_sentences']} of {rw['total_sentences']} "
            f"sentences edited.")
        add("")
        add(rw["text"])
        if rw["unresolved"]:
            add("")
            add("STILL NEEDS A HUMAN LOOK (not auto-applied — a judgment call):")
            for i, u in enumerate(rw["unresolved"], 1):
                prefix = f"  {i}. "
                wrapped = _wrap(u, 68, " " * len(prefix))
                wrapped[0] = prefix + wrapped[0].lstrip()
                lines.extend(wrapped)
        add("")
        add("This is a mechanical, rule-based edit — not a paraphrase and "
            "not a claim that the result reads better. Read it before "
            "keeping it. Per-sentence diff (original -> edited, with the "
            "reason for each change) is in --json output under \"rewrite\".")

    if "ai_probability" in report:
        ai = report["ai_probability"]
        add("")
        add("=" * 72)
        add("AI-WRITING PROBABILITY (learned classifier — read the caveat)")
        add("=" * 72)
        add("")
        add(f"  P(AI-written) = {ai['ai_probability']:.2%}"
            f"  ({ai['n_windows']} window(s) averaged)")
        add("")
        add(CLASSIFIER_DISCLAIMER)

    if show_disclaimer:
        add("")
        add("-" * 72)
        add(DISCLAIMER)

    return "\n".join(lines)
