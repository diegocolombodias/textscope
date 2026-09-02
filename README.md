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

## Using it fairly

If you use this on student work, the output is a reason to open a
conversation, never a finding. Ask the student to walk you through the
argument, explain a source choice, or show how the draft developed.
That conversation is evidence. A z-score is not.
