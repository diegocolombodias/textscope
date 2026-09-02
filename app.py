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


def _heat_colour(nll: float | None) -> str:
    """Low surprisal -> warm highlight. None -> no highlight."""
    if nll is None:
        return "transparent"
    # 0 nats = perfectly predictable, 6+ nats = very surprising
    t = max(0.0, min(1.0, 1.0 - (nll / 6.0)))
    alpha = 0.10 + 0.45 * t
    return f"rgba(220, 120, 40, {alpha:.2f})"


def _flag_heat_colour(n_flags: int) -> str:
    """Fallback heat map when no LM is loaded: colour by stylometric
    flag count on the sentence (tricolon, scaffold phrase, lexicon hit,
    stacked hedges). More flags -> stronger highlight. This is a count
    of pattern matches, not a severity score — it says "several
    candidate markers landed on this sentence", nothing more."""
    if n_flags <= 0:
        return "transparent"
    alpha = min(0.55, 0.16 + 0.13 * n_flags)
    return f"rgba(220, 120, 40, {alpha:.2f})"


def analyze(text: str, lang: str) -> tuple[str, str]:
    if not text.strip():
        return "<p>Paste some text.</p>", ""

    rep = build_report(text, lang=lang, scorer=SCORER)

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
            f"<sup style='color:#b45309;font-size:0.75em'>&nbsp;{html.escape(tip)}</sup>"
            if tip else ""
        )
        blocks.append(
            f"<span style='background:{bg};padding:2px 1px;border-radius:3px'>"
            f"{html.escape(e['text'])}</span>{badge} "
        )

    highlighted = (
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
    return highlighted, "\n".join(lines)


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
        heat_note = ("Highlight intensity = surprisal (how predictable the "
                     "sentence is to the local model). Darker is not "
                     "\"more AI\" — it's lower-surprisal for *this* model.")
    else:
        heat_note = ("Highlight intensity = number of stylometric flags on "
                     "the sentence (no model loaded, style-only mode). "
                     "Darker just means more pattern matches landed there — "
                     "read those first, don't take the colour as a verdict.")

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

        btn.click(analyze, inputs=[inp, lang], outputs=[out_html, out_md])

    # share=False: no tunneling through Gradio's servers. Binding is
    # explicit via --host (loopback by default, 0.0.0.0 in a container).
    demo.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
