# TextScope

Local, private analysis of prose style. Runs entirely on your machine.

## What it does

Two independent layers:

1. **Stylometry** (no model needed) — sentence-length variance, tricolon
   density, scaffolding phrases, LLM-frequent lexicon, stacked hedges,
   em-dash rate, paragraph-opener repetition.

2. **Surprisal** (optional local LM) — per-token negative log-likelihood,
   perplexity, token rank, predictive entropy. This is the family of
   features commercial detectors use.

3. **Calibration** — the part that makes the other two interpretable.
   You build a reference corpus from writing whose provenance you know,
   and the tool reports standard deviations from *that* corpus.

## What it does not do

It does not output a probability that text was written by AI. That
refusal is deliberate, not an unfinished feature.

Detectors measure predictability, not authorship. Formal academic prose
is predictable by construction. So is writing by authors working in a
second language — the documented failure mode is severe, with reported
false-positive rates above 60% on TOEFL essays by non-native speakers
against near-zero on native writing. A tool built on these features
inherits that bias no matter how it is packaged. Emitting a percentage
would launder a stylistic measurement into a claim about a person.

## Install

    pip install -r requirements.txt          # stylometry only
    pip install torch transformers           # + surprisal
    pip install gradio                       # + web UI

## Get a model (once, then fully offline)

    pip install huggingface_hub
    huggingface-cli download gpt2-large --local-dir ./models/gpt2-large
    # Portuguese:
    huggingface-cli download pierreguillou/gpt2-small-portuguese \
        --local-dir ./models/gpt2-pt

## Use

    python cli.py analyze paper.txt --lang en
    python cli.py analyze paper.txt --model ./models/gpt2-large
    python cli.py calibrate "corpus/*.txt" --out reference.json
    python cli.py analyze draft.txt --reference reference.json
    python app.py --model ./models/gpt2-large      # http://127.0.0.1:7860

## Reading the highlight colors (web UI)

With a model loaded (`app.py --model ...`), each sentence in the
annotated-text panel gets a background color and a small `ppl=N.NN` badge.
Both come from the same number, so here is what it means and how to read
the color.

**What `ppl` is.** Perplexity — `exp(average per-token negative
log-likelihood)` for that sentence, under the local model. Loosely: "how
surprised the model was, on average, predicting each word given what came
before."

* **Low ppl** (a few units, e.g. `ppl=8.2`) — the sentence was highly
  predictable to the model. Generic, low-information phrasing scores low
  here, which is also the shape of most LLM-generated prose — that overlap
  is the entire reason this feature family exists.
* **High ppl** (dozens to hundreds, e.g. `ppl=101.95`) — the sentence
  surprised the model. Technical jargon, an unusual name, a formula badly
  extracted from a PDF, or a switch in register can all push this up.

**The color is diverging, not a single intensity, on purpose.**
Predictability has a genuine middle — "unremarkable, about as surprising as
the rest of the text" — with two distinct extremes on either side of it, so
a two-hue scale with a neutral midpoint is the correct encoding (a single
hue getting darker only makes sense for a plain 0-to-many magnitude, which
this isn't).

| Color | Meaning | Nats (NLL) |
|---|---|---|
| 🔴 red, saturated | very predictable — the flat, low-surprisal signal this tool measures | ~0 |
| pale red → gray | leaning predictable, mild | 0–3 |
| gray, faint | unremarkable — nothing to see here | ~3 |
| gray → pale blue | leaning surprising, mild | 3–6 |
| 🔵 blue, saturated | very surprising to the model | ~6+ |

Color *intensity* also carries information: it fades toward transparent
near the gray midpoint and strengthens toward either pole, so the page
draws your eye to the sentences actually worth a second look instead of
tinting everything uniformly.

**Without a loaded model** (stylometry-only mode), the highlight falls back
to a single orange hue whose *intensity* (not hue) tracks how many
stylometric flags landed on that sentence — a plain count, not a spectrum,
so there's no second color to diverge toward.

**Same caveat as everywhere else in this tool:** the color tells you where
to look, not what you'll find when you get there. Red is not "AI." Blue is
not "definitely human." A dense related-work paragraph can turn blue for
the same reason a second-language author can turn red — see *What it does
not do*, above.

## Using it fairly

If you use this on student work, the output is a reason to open a
conversation, never a finding. Ask the student to walk you through the
argument, explain a source choice, or show how the draft developed.
That conversation is evidence. A z-score is not.
