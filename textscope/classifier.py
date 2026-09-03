"""
Supervised black-box classifier: P(this text is AI-written).

This module is a deliberate departure from the rest of TextScope. Every
other module in this package refuses to output a probability of AI
authorship — see README.md, "What it does not do" — because no such
number can be computed honestly from stylometric or surprisal features
alone. This module exists anyway, built on request, to let you work with
the same kind of supervised approach commercial detectors (Turnitin,
GPTZero, ...) actually ship: a transformer fine-tuned on labeled
human/AI text to output a single probability per document.

Read this before trusting anything it outputs:

  * It inherits every false-positive risk the rest of this project's
    README warns about — non-native English, formal or technical prose,
    unusual domains — and it hides that risk behind one opaque number
    instead of an inspectable z-score against a corpus you chose.
  * Its training data is small and mixed-provenance relative to what a
    commercial vendor uses: the public HC3 corpus (real human answers vs
    real ChatGPT answers, general-domain) plus the 15 papers in
    reference_corpus/ (human, academic) plus 219 academic-style
    paragraphs actually written by three different LLMs (Claude, Gemini,
    Kimi — see reference_corpus/ai_samples_*.jsonl) as the "AI writing in
    the papers register" class. This replaced an earlier version of this
    file that used ~40 hand-written stand-in paragraphs instead of real
    model output; a saturation test (documented in this project's
    capacitação report) showed the hand-written-stand-in classifier had
    essentially zero resolution within the formal-academic register,
    returning ~94.6% for every academic paragraph tested regardless of
    content or phrasing. Real LLM output should generalize better, but
    it's still three models' worth of samples, not the "two decades of
    real student writing and dozens of LLM prompting strategies" Turnitin's
    published whitepaper describes as its training corpus — nowhere near
    that scale or diversity.
  * It has not been independently validated. `train_classifier()` prints
    and saves held-out recall and false-positive rate — by the same two
    metrics Turnitin's whitepaper uses, deliberately not "accuracy" (see
    its docstring for why) — READ THOSE NUMBERS before trusting the
    model on anything, and re-check them on text similar to what you
    actually plan to analyze, ideally by rebuilding the academic-AI
    portion of the training data with real LLM output before relying on
    this for anything that matters.
  * English only. Every training source (HC3, the reference-corpus
    papers, the hand-written academic samples) is English; there is no
    language check at inference time, so scoring Portuguese text will
    silently return a meaningless number instead of an error. Turnitin's
    own whitepaper states the same English-only limitation for the same
    reason: optimize one language before attempting more.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_BASE_MODEL = "roberta-base"
MAX_LENGTH = 256


def _chunk_words(text: str, chunk_words: int = 220) -> list[str]:
    """Split long prose into ~chunk_words-word pieces on paragraph
    boundaries where possible. Used to turn the (long) reference-corpus
    papers into training-example-sized pieces."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    count = 0
    for p in paragraphs:
        words = p.split()
        if count + len(words) > chunk_words and buf:
            chunks.append(" ".join(buf))
            buf, count = [], 0
        buf.extend(words)
        count += len(words)
        if count >= chunk_words:
            chunks.append(" ".join(buf))
            buf, count = [], 0
    if buf:
        chunks.append(" ".join(buf))
    return [c for c in chunks if len(c.split()) >= 40]


def load_training_examples(
    hc3_path: str,
    reference_corpus_dir: str,
    ai_samples_path: str,
    max_hc3_per_class: Optional[int] = 15000,
    seed: int = 0,
) -> list[dict]:
    """
    Assemble (text, label) examples from three sources, each row also
    tagged with "source" so train_classifier can split before
    oversampling instead of after (see its docstring for why that
    order matters). label: 0 = human, 1 = AI.

      * HC3 ("Hello-SimpleAI/HC3", all.jsonl): real human vs real ChatGPT
        answers to the same questions across several domains (open QA,
        finance, medicine, ELI5, wiki/CS-AI). The primary signal — real
        text on both sides, not a stand-in.
      * reference_corpus/txt/*.txt: the 15 real arXiv papers used
        elsewhere in this project, chunked into ~220-word pieces, as
        additional human/academic-register examples. HC3 alone skews
        conversational; this adds the dense academic register TextScope
        is actually used on.
      * ai_samples_path (reference_corpus/ai_academic_samples.jsonl):
        219 real academic-style paragraphs from three different LLMs
        (Claude, Gemini, Kimi — see reference_corpus/ai_samples_*.jsonl,
        combined here) — see the module docstring for their limits.
        No oversampling happens here anymore — see train_classifier.
    """
    examples: list[dict] = []
    rng = random.Random(seed)

    with open(hc3_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    rng.shuffle(rows)

    human_count = ai_count = 0
    for row in rows:
        for ans in row.get("human_answers") or []:
            ans = " ".join(ans.split())
            if len(ans.split()) < 20:
                continue
            if max_hc3_per_class is not None and human_count >= max_hc3_per_class:
                continue
            examples.append({"text": ans, "label": 0, "source": "hc3"})
            human_count += 1
        for ans in row.get("chatgpt_answers") or []:
            ans = " ".join(ans.split())
            if len(ans.split()) < 20:
                continue
            if max_hc3_per_class is not None and ai_count >= max_hc3_per_class:
                continue
            examples.append({"text": ans, "label": 1, "source": "hc3"})
            ai_count += 1

    n_paper_chunks = 0
    for path in sorted(Path(reference_corpus_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for chunk in _chunk_words(text):
            examples.append({"text": chunk, "label": 0, "source": "paper"})
            n_paper_chunks += 1

    n_academic_ai = 0
    with open(ai_samples_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            examples.append({
                "text": row["text"], "label": 1, "source": "academic_ai",
            })
            n_academic_ai += 1

    print(
        f"Training examples: {human_count} HC3-human, {ai_count} HC3-AI, "
        f"{n_paper_chunks} reference-paper chunks (human), "
        f"{n_academic_ai} academic-AI (oversampled after the split, not "
        f"here) (total {len(examples)})"
    )

    rng.shuffle(examples)
    return examples


def train_classifier(
    hc3_path: str,
    reference_corpus_dir: str,
    ai_samples_path: str,
    out_dir: str,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = 2,
    batch_size: int = 16,
    max_hc3_per_class: Optional[int] = 15000,
    device: str = "cuda",
    learning_rate: Optional[float] = None,
    adam_epsilon: Optional[float] = None,
    academic_ai_oversample: int = 2,
) -> dict:
    """
    Fine-tune `base_model` as a binary human/AI sequence classifier and
    save it to `out_dir`. Returns the held-out evaluation metrics dict
    (also written to `out_dir/eval_metrics.json`).

    Deliberately reports Recall and False-Positive Rate, not accuracy —
    see Turnitin's own whitepaper for why accuracy is a bad metric here
    (a classifier that always predicts "human" scores ~50% accuracy on a
    balanced set here and would be useless).

    `academic_ai_oversample` is applied to the *training* split only,
    after the train/val/test split — never before. Oversampling before
    splitting (the original implementation) put literal duplicate copies
    of the same paragraph on both sides of the split, so "held-out"
    recall partly measured memorization of a paragraph the model had
    already seen verbatim in training, not generalization. That bug
    produced wildly unstable eval numbers as the oversample factor
    changed (recall/fpr collapsing to 0/0 at one setting and 1/1 at
    another) before it was found and fixed here.
    """
    import numpy as np
    import torch
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    examples = load_training_examples(
        hc3_path, reference_corpus_dir, ai_samples_path,
        max_hc3_per_class=max_hc3_per_class,
    )
    texts = [e["text"] for e in examples]
    labels = [e["label"] for e in examples]
    sources = [e["source"] for e in examples]

    (train_texts, temp_texts, train_labels, temp_labels,
     train_sources, _temp_sources) = train_test_split(
        texts, labels, sources, test_size=0.2, random_state=0,
        stratify=labels,
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=0,
        stratify=temp_labels,
    )

    if academic_ai_oversample > 1:
        extra_texts, extra_labels = [], []
        for t, lbl, src in zip(train_texts, train_labels, train_sources):
            if src == "academic_ai":
                extra_texts.extend([t] * (academic_ai_oversample - 1))
                extra_labels.extend([lbl] * (academic_ai_oversample - 1))
        train_texts = train_texts + extra_texts
        train_labels = train_labels + extra_labels

    print(f"Split: {len(train_texts)} train (oversample x"
          f"{academic_ai_oversample} applied here) / {len(val_texts)} "
          f"val / {len(test_texts)} test")

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    class _TextDataset(torch.utils.data.Dataset):
        def __init__(self, texts: list[str], labels: list[int]):
            self.enc = tokenizer(
                texts, truncation=True, max_length=MAX_LENGTH, padding=False,
            )
            self.labels = labels

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> dict:
            item = {k: v[idx] for k, v in self.enc.items()}
            item["labels"] = self.labels[idx]
            return item

    train_ds = _TextDataset(train_texts, train_labels)
    val_ds = _TextDataset(val_texts, val_labels)
    test_ds = _TextDataset(test_texts, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2
    )

    from transformers import DataCollatorWithPadding
    collator = DataCollatorWithPadding(tokenizer)

    def _metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        return {"recall": recall, "fpr": fpr}

    # DeBERTa-v2/v3's disentangled attention is numerically unstable
    # under mixed precision: fp16 hits "Attempting to unscale FP16
    # gradients" (its loss-scaling GradScaler chokes on some parameter
    # shape in the relative-position bias); bf16 avoids that crash but
    # trades away enough mantissa precision to silently diverge instead
    # (eval_loss -> NaN, a collapsed classifier). This is a known
    # instability in that architecture, not something specific to this
    # dataset — train it in plain fp32. The backbone is small (~86M
    # params) so the extra cost is a couple of minutes, not a problem.
    is_deberta = "deberta" in base_model.lower()
    use_mixed_precision = device == "cuda" and not is_deberta

    if learning_rate is None:
        # DeBERTa-v2/v3's disentangled attention is a known-fragile
        # architecture to fine-tune: grad_norm reliably blows up to NaN
        # partway through training, independent of fp16 vs bf16 vs
        # fp32. Warmup + a halved LR (below) is the standard first fix
        # cited for this — confirmed NOT sufficient on its own here, so
        # treat 1e-5 as a starting point to combine with a lower
        # --learning-rate and/or --adam-epsilon override, not a fix by
        # itself. RoBERTa has no such issue and keeps the original rate.
        learning_rate = 1e-5 if is_deberta else 2e-5
    if adam_epsilon is None:
        adam_epsilon = 1e-8  # transformers' own default

    # This transformers version's TrainingArguments dropped the
    # warmup_ratio convenience kwarg (warmup_steps only) — compute the
    # step count for the same ~6% warmup by hand.
    import math
    steps_per_epoch = math.ceil(len(train_texts) / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(0.06 * total_steps))

    args = TrainingArguments(
        output_dir=str(Path(out_dir) / "_checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        learning_rate=learning_rate,
        adam_epsilon=adam_epsilon,
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,
        weight_decay=0.01,
        fp16=use_mixed_precision,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=_metrics,
    )
    trainer.train()

    test_metrics = trainer.evaluate(test_ds)
    print("Held-out test metrics:", test_metrics)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (Path(out_dir) / "eval_metrics.json").write_text(
        json.dumps(test_metrics, indent=2)
    )
    (Path(out_dir) / "TRAINING_NOTE.md").write_text(
        "This checkpoint was fine-tuned from {base} on HC3 + "
        "reference_corpus + hand-written academic-AI samples. See "
        "textscope/classifier.py's module docstring for the honest "
        "limitations before trusting its output.\n\n"
        "Held-out test metrics: {metrics}\n".format(
            base=base_model, metrics=json.dumps(test_metrics, indent=2)
        )
    )
    return test_metrics


@dataclass
class ClassifierResult:
    ai_probability: float
    n_windows: int
    window_probabilities: list[float]

    def as_dict(self) -> dict:
        return {
            "ai_probability": round(self.ai_probability, 4),
            "n_windows": self.n_windows,
            "window_probabilities": [round(p, 4) for p in self.window_probabilities],
        }


class ClassifierScorer:
    """
    Loads a fine-tuned checkpoint from `train_classifier` and scores text.

    Long documents are split into ~220-word windows (mirroring the
    sliding-window aggregation Turnitin's whitepaper describes) and the
    document-level probability is the mean of the per-window
    probabilities — a plain average, not Turnitin's stride-and-threshold
    scheme, since this checkpoint has no equivalent calibration data to
    tune such a scheme against.
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

    def predict_proba(self, text: str) -> ClassifierResult:
        torch = self._torch
        windows = _chunk_words(text) or [text]

        probs: list[float] = []
        with torch.no_grad():
            for w in windows:
                enc = self.tokenizer(
                    w, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
                ).to(self.device)
                logits = self.model(**enc).logits[0]
                p_ai = torch.softmax(logits, dim=-1)[1].item()
                probs.append(p_ai)

        return ClassifierResult(
            ai_probability=sum(probs) / len(probs),
            n_windows=len(probs),
            window_probabilities=probs,
        )
