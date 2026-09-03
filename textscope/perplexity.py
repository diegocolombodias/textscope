"""
Token-level surprisal from a locally hosted causal language model.

This is the component that approximates what commercial detectors do:
text that a language model finds highly predictable (low surprisal, low
variance in surprisal) is what those tools score as machine-written.

The model never leaves your machine. Nothing is uploaded.

IMPORTANT — read before trusting any number this produces:

  * Surprisal is measured *relative to the scoring model*. A different
    model gives different numbers for the same text. There is no
    absolute scale.
  * Formal academic prose is intrinsically low-surprisal. So is prose by
    writers working in a second language. Both are penalised by this
    method for reasons that have nothing to do with authorship.
  * The only defensible use is comparative: score a candidate text
    against a reference corpus of text you KNOW the same author wrote,
    under the same model. See calibration.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TokenScore:
    token: str
    nll: float          # negative log-likelihood, nats
    rank: int           # rank of the observed token in the predicted dist.
    entropy: float      # entropy of the predictive distribution


@dataclass
class PerplexityReport:
    tokens: list[TokenScore]
    mean_nll: float
    perplexity: float
    nll_stdev: float           # "burstiness" in the surprisal domain
    median_rank: float
    top10_rate: float          # share of tokens ranked in the model's top 10
    mean_entropy: float

    def as_dict(self) -> dict:
        return {
            "mean_nll_nats": round(self.mean_nll, 4),
            "perplexity": round(self.perplexity, 2),
            "nll_stdev": round(self.nll_stdev, 4),
            "median_token_rank": self.median_rank,
            "share_in_model_top10": round(self.top10_rate, 4),
            "mean_predictive_entropy": round(self.mean_entropy, 4),
        }


class LocalScorer:
    """
    Wraps a Hugging Face causal LM held entirely on local disk.

    Suggested models (all run on CPU, none require an API key):
      English   : "gpt2-large"  (~3 GB)  or "gpt2"      (~500 MB)
      Portuguese: "pierreguillou/gpt2-small-portuguese"
      Multiling.: "bigscience/bloom-560m"

    Download once with:
        huggingface-cli download gpt2-large --local-dir ./models/gpt2-large
    then pass model_path="./models/gpt2-large" and run fully offline.
    """

    def __init__(
        self,
        model_path: str = "gpt2",
        device: str = "cpu",
        max_length: Optional[int] = None,
        dtype: Optional[str] = None,
    ) -> None:
        # Imported lazily so the stylometric half of the tool works
        # without torch installed.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        load_kwargs = {}
        if dtype:
            # fp16/bf16 for multi-billion-parameter models — fp32 doubles
            # VRAM for no accuracy benefit on a scoring-only (no gradient)
            # workload. Irrelevant for small models like gpt2-large.
            load_kwargs["torch_dtype"] = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.to(device)
        self.model.eval()
        self.max_length = max_length or getattr(
            self.model.config, "n_positions",
            getattr(self.model.config, "max_position_embeddings", 1024),
        )

    def score(self, text: str) -> PerplexityReport:
        torch = self._torch

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc["input_ids"].to(self.device)

        if input_ids.shape[1] < 2:
            raise ValueError("Text too short to score (need >= 2 tokens).")

        with torch.no_grad():
            logits = self.model(input_ids).logits

        # Predict token t from context < t.
        shift_logits = logits[0, :-1, :]
        shift_labels = input_ids[0, 1:]

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_nll = -log_probs[range(len(shift_labels)), shift_labels]

        # Rank of the observed token among all vocabulary items.
        sorted_idx = torch.argsort(shift_logits, dim=-1, descending=True)
        ranks = (sorted_idx == shift_labels.unsqueeze(-1)).nonzero()[:, 1] + 1

        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)

        toks = self.tokenizer.convert_ids_to_tokens(shift_labels)

        scores = [
            TokenScore(
                token=t,
                nll=float(n),
                rank=int(r),
                entropy=float(e),
            )
            for t, n, r, e in zip(toks, token_nll, ranks, entropy)
        ]

        nll_arr = np.array([s.nll for s in scores])
        rank_arr = np.array([s.rank for s in scores])
        ent_arr = np.array([s.entropy for s in scores])

        return PerplexityReport(
            tokens=scores,
            mean_nll=float(nll_arr.mean()),
            perplexity=float(math.exp(nll_arr.mean())),
            nll_stdev=float(nll_arr.std()),
            median_rank=float(np.median(rank_arr)),
            top10_rate=float((rank_arr <= 10).mean()),
            mean_entropy=float(ent_arr.mean()),
        )

    def score_sentences(self, sentences: list[str]) -> list[PerplexityReport]:
        """Per-sentence scoring, for locating the low-surprisal regions."""
        out = []
        for s in sentences:
            try:
                out.append(self.score(s))
            except ValueError:
                out.append(None)
        return out
