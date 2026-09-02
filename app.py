#!/usr/bin/env python3
"""
TextScope web UI — runs on localhost, sends nothing anywhere.

    pip install gradio
    python app.py                      # stylometry only
    python app.py --model ./models/gpt2-large

Then open http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr  # noqa: E402

from textscope.report import build_report, DISCLAIMER  # noqa: E402
from textscope import calibration  # noqa: E402

SCORER = None
REFERENCE = None


BADGE_INK_LIGHT = "#52514e"  # secondary ink, light surface
BADGE_INK_DARK = "#c3c2b7"   # secondary ink, dark surface — Gradio defaults
                              # to the OS/browser dark theme, so the badge
                              # needs its own step there too (palette.md),
                              # not just a lighter alpha of the light one.

# Diverging heat-map poles for per-sentence surprisal (the blue<->red
# diverging pair, light-surface values). Predictability is a spectrum with
# a genuine midpoint ("unremarkable"), so it gets two opposite hues, not
# one hue at increasing strength — see README.md § "Reading the highlight
# colors" for the full explanation.
_POLE_PREDICTABLE = (0xE3, 0x49, 0x48)  # #e34948 red  — low surprisal
_POLE_NEUTRAL = (0xF0, 0xEF, 0xEC)      # #f0efec gray — unremarkable
_POLE_SURPRISING = (0x2A, 0x78, 0xD6)   # #2a78d6 blue — high surprisal

# Single-hue sequential ramp for the no-model fallback (flag count is a
# magnitude, zero to many, with no "opposite" pole to diverge against).
_FLAG_HUE = (0xEB, 0x68, 0x34)  # #eb6834 orange


def _lerp_rgb(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _heat_colour(nll: float | None) -> str:
    """Diverging surprisal heat map: red = predictable, blue = surprising.

    0 nats = perfectly predictable to the local model — the low-surprisal,
    flat-affect end this tool is built to surface. 6+ nats = highly
    unpredictable — dense jargon, rare phrasing, a language switch, none
    of which is evidence of anything by itself (see the disclaimer). The
    midpoint (3 nats) is unremarkable and fades toward the page background
    rather than competing for attention with the two poles.
    """
    if nll is None:
        return "transparent"
    t = max(0.0, min(1.0, nll / 6.0))  # 0 = predictable, 1 = surprising
    if t <= 0.5:
        rgb = _lerp_rgb(_POLE_PREDICTABLE, _POLE_NEUTRAL, t / 0.5)
    else:
        rgb = _lerp_rgb(_POLE_NEUTRAL, _POLE_SURPRISING, (t - 0.5) / 0.5)
    # Fade toward transparent at the midpoint (nothing to see), opaque at
    # either pole (something worth reading) — distance from center drives
    # intensity so the eye is drawn to extremes, not to the whole page.
    alpha = 0.08 + 0.42 * (abs(t - 0.5) * 2)
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha:.2f})"


def _flag_heat_colour(n_flags: int) -> str:
    """Fallback heat map when no LM is loaded: colour by stylometric
    flag count on the sentence (tricolon, scaffold phrase, lexicon hit,
    stacked hedges). More flags -> stronger highlight, one hue, since
    flag count has no opposite pole to diverge against — it is a plain
    magnitude. This is a count of pattern matches, not a severity score —
    it says "several candidate markers landed on this sentence", nothing
    more."""
    if n_flags <= 0:
        return "transparent"
    alpha = min(0.55, 0.16 + 0.13 * n_flags)
    r, g, b = _FLAG_HUE
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


def analyze(text: str, lang: str) -> tuple[str, str, str]:
    if not text.strip():
        return "<p>Paste some text.</p>", "", ""

    rep = build_report(text, lang=lang, scorer=SCORER, reference=REFERENCE)

    blocks = []
    for e in rep["sentences"]:
        if "mean_nll" in e:
            bg = _heat_colour(e["mean_nll"])
        else:
            bg = _flag_heat_colour(len(e["flags"]))
        tip = "; ".join(e["flags"]) if e["flags"] else ""
        if "perplexity" in e:
            tip = (tip + " | " if tip else "") + f"ppl={e['perplexity']}"
        badge = (
            f"<sup class='ts-badge' style='font-size:0.75em'>&nbsp;{html.escape(tip)}</sup>"
            if tip else ""
        )
        blocks.append(
            f"<span style='background:{bg};padding:2px 1px;border-radius:3px'>"
            f"{html.escape(e['text'])}</span>{badge} "
        )

    highlighted = (
        f"<style>.ts-badge{{color:{BADGE_INK_LIGHT}}}"
        f"@media (prefers-color-scheme: dark){{.ts-badge{{color:{BADGE_INK_DARK}}}}}</style>"
        "<div style='line-height:2.0;font-family:Georgia,serif;font-size:15px'>"
        + "".join(blocks)
        + "</div>"
    )

    lines = ["### Document features", ""]
    for k, v in rep["style"].items():
        if isinstance(v, dict):
            if v:
                lines.append(f"- **{k}**: "
                             + ", ".join(f"{kk} ({vv})" for kk, vv in v.items()))
        else:
            lines.append(f"- **{k}**: {v}")

    doc_suggestions = rep.get("document_suggestions", [])
    if doc_suggestions:
        lines += ["", "### Document-level suggestions", ""]
        lines += [f"{i}. {s}" for i, s in enumerate(doc_suggestions, 1)]

    flagged = [e for e in rep["sentences"] if e["flags"]]
    if flagged:
        lines += ["", "### Per-sentence suggestions", ""]
        for e in flagged:
            lines.append(f"**[s{e['index']}]** {e['text']}")
            for s in e.get("suggestions", []):
                lines.append(f"- {s}")
            lines.append("")

    if "surprisal" in rep:
        lines += ["", "### Surprisal (local model)", ""]
        for k, v in rep["surprisal"].items():
            lines.append(f"- **{k}**: {v}")

    if REFERENCE is not None:
        feats = {k: float(v) for k, v in rep["style"].items()
                 if isinstance(v, (int, float))}
        if "surprisal" in rep:
            feats.update({k: float(v) for k, v in rep["surprisal"].items()})
        z = calibration.z_scores(feats, REFERENCE)
        lines += ["", f"### vs reference corpus "
                      f"({REFERENCE.n_documents} docs)", ""]
        lines += calibration.interpret(z)

    lines += ["", "---", "", DISCLAIMER.replace("\n", "  \n")]

    rw = rep["rewrite"]
    rw_lines = [
        f"**{rw['changed_sentences']} of {rw['total_sentences']} "
        f"sentences edited.**",
        "",
        rw["text"],
    ]
    if rw["unresolved"]:
        rw_lines += ["", "**Still needs a human look "
                          "(not auto-applied — a judgment call):**", ""]
        rw_lines += [f"{i}. {u}" for i, u in enumerate(rw["unresolved"], 1)]
    rw_lines += ["", "---", "",
                 "*Mechanical, rule-based edit — not a paraphrase, and not "
                 "a claim that the result reads better. Read it before "
                 "keeping it.*"]

    return highlighted, "\n".join(lines), "\n".join(rw_lines)


def main() -> None:
    global SCORER, REFERENCE

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="path to a local HF causal LM directory")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (use 0.0.0.0 inside a container)")
    args = ap.parse_args()

    if args.model:
        from textscope.perplexity import LocalScorer
        print(f"Loading {args.model} ...")
        SCORER = LocalScorer(model_path=args.model, device=args.device)

    if args.reference:
        REFERENCE = calibration.ReferenceStats.load(args.reference)

    if SCORER is not None:
        heat_note = (
            "🔴 **red** = predictable to the local model (low surprisal — "
            "the flat, low-variance signal this tool is built to surface). "
            "🔵 **blue** = unpredictable (dense jargon, rare phrasing, a "
            "language switch — not evidence of anything by itself). "
            "Faint/gray = unremarkable, near the middle. Neither colour is "
            "\"more AI\" on its own — see the disclaimer, and "
            "README.md § *Reading the highlight colors* for the full "
            "explanation and the reasoning behind it."
        )
    else:
        heat_note = ("Highlight intensity (single orange hue) = number of "
                     "stylometric flags on the sentence (no model loaded, "
                     "style-only mode). Darker just means more pattern "
                     "matches landed there — read those first, don't take "
                     "the colour as a verdict.")

    with gr.Blocks(title="TextScope") as demo:
        gr.Markdown(
            "# TextScope\n"
            "Local prose analysis. Nothing leaves this machine.\n\n"
            "*This tool locates passages worth reading closely. "
            "It does not determine authorship.*\n\n"
            f"**Reading the highlights:** {heat_note}"
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Textbox(lines=18, label="Text",
                                 placeholder="Paste a paragraph or a paper...")
                lang = gr.Radio(["en", "pt"], value="en", label="Language")
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=1):
                out_html = gr.HTML(label="Annotated text")
                out_md = gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Suggested rewrite")
                out_rewrite = gr.Markdown()

        btn.click(analyze, inputs=[inp, lang],
                  outputs=[out_html, out_md, out_rewrite])

    # share=False: no tunneling through Gradio's servers. Binding is
    # explicit via --host (loopback by default, 0.0.0.0 in a container).
    demo.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
