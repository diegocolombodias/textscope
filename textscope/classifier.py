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
    reference_corpus/ (human, academic) plus a few dozen academic-style
    paragraphs this project's own assistant wrote as a stand-in for
    "AI writing in the papers register" — there was no API access to a
    modern commercial LLM at training time to generate that class
    properly. Turnitin's published whitepaper describes a training
    corpus spanning two decades of real student writing and dozens of
    LLM prompting strategies; this is nowhere near that scale.
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
    academic_ai_oversample: int = 8,
    seed: int = 0,
) -> list[dict]:
    """
    Assemble (text, label) examples from three sources. label: 0 = human,
    1 = AI.

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
        hand-written academic-style AI paragraphs — see the module
        docstring for why these exist and their limits. Oversampled
        (default x8) since there are only a few dozen of them against
        thousands of HC3 examples per class; this is duplication, not
        augmentation, and is not a substitute for real held-out variety.
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
            examples.append({"text": ans, "label": 0})
            human_count += 1
        for ans in row.get("chatgpt_answers") or []:
            ans = " ".join(ans.split())
            if len(ans.split()) < 20:
                continue
            if max_hc3_per_class is not None and ai_count >= max_hc3_per_class:
                continue
            examples.append({"text": ans, "label": 1})
            ai_count += 1

    n_paper_chunks = 0
    for path in sorted(Path(reference_corpus_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for chunk in _chunk_words(text):
            examples.append({"text": chunk, "label": 0})
            n_paper_chunks += 1

    academic_ai: list[dict] = []
    with open(ai_samples_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            academic_ai.append({"text": row["text"], "label": 1})
    examples.extend(academic_ai * academic_ai_oversample)

    print(
        f"Training examples: {human_count} HC3-human, {ai_count} HC3-AI, "
        f"{n_paper_chunks} reference-paper chunks (human), "
        f"{len(academic_ai)} academic-AI x{academic_ai_oversample} "
        f"oversample = {len(academic_ai) * academic_ai_oversample} "
        f"(total {len(examples)})"
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
) -> dict:
    """
    Fine-tune `base_model` as a binary human/AI sequence classifier and
    save it to `out_dir`. Returns the held-out evaluation metrics dict
    (also written to `out_dir/eval_metrics.json`).

    Deliberately reports Recall and False-Positive Rate, not accuracy —
    see Turnitin's own whitepaper for why accuracy is a bad metric here
    (a classifier that always predicts "human" scores ~50% accuracy on a
    balanced set here and would be useless).
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

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=0, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=0,
        stratify=temp_labels,
    )
    print(f"Split: {len(train_texts)} train / {len(val_texts)} val / "
          f"{len(test_texts)} test")

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

    args = TrainingArguments(
        output_dir=str(Path(out_dir) / "_checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=(device == "cuda"),
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
