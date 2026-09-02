"""
Calibration against a reference corpus.

This is the most important module in the package, and the one commercial
detectors do not give you.

A raw perplexity number means nothing. 'Perplexity 18.4' is not high or
low in the abstract — it is high or low *relative to a population*.
Vendors hide their reference population, which is why their thresholds
transfer badly to academic prose and to second-language writers.

Here you build the reference population yourself:

  * To review student work: a corpus of past work from the same cohort,
    same assignment type, written before generative models were
    available. Then a flag means "unusual for this cohort", which is a
    claim you can actually defend.
  * To review your own drafts: your own published papers. Then a flag
    means "this paragraph does not sound like your other writing".

Both are honest questions. "Is this AI?" is not a question this tool,
or any tool, can answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass
class ReferenceStats:
    n_documents: int
    feature_means: dict[str, float]
    feature_stdevs: dict[str, float]
    model_id: str
    note: str = ""

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceStats":
        return cls(**json.loads(Path(path).read_text()))


def build_reference(
    feature_dicts: Iterable[dict[str, float]],
    model_id: str,
    note: str = "",
) -> ReferenceStats:
    """Aggregate per-document feature dicts into a reference distribution."""
    rows = list(feature_dicts)
    if len(rows) < 15:
        raise ValueError(
            f"Only {len(rows)} reference documents supplied. Below roughly "
            "15 the standard deviations are too unstable for z-scores to "
            "mean anything. Collect more, or do not calibrate at all."
        )

    keys = sorted(set().union(*(set(r) for r in rows)))
    means, stdevs = {}, {}
    for k in keys:
        vals = np.array([float(r[k]) for r in rows if k in r], dtype=float)
        if vals.size < 2:
            continue
        means[k] = float(vals.mean())
        stdevs[k] = float(vals.std(ddof=1))

    return ReferenceStats(
        n_documents=len(rows),
        feature_means=means,
        feature_stdevs=stdevs,
        model_id=model_id,
        note=note,
    )


def z_scores(
    features: dict[str, float],
    reference: ReferenceStats,
) -> dict[str, Optional[float]]:
    """Standardise a document's features against the reference corpus."""
    out: dict[str, Optional[float]] = {}
    for k, v in features.items():
        mu = reference.feature_means.get(k)
        sd = reference.feature_stdevs.get(k)
        if mu is None or sd is None or sd == 0:
            out[k] = None
        else:
            out[k] = (float(v) - mu) / sd
    return out


def interpret(z: dict[str, Optional[float]]) -> list[str]:
    """
    Turn z-scores into cautious prose. Deliberately refuses to output a
    single composite score or a probability — there is no principled way
    to produce one, and a single number invites exactly the misuse this
    package exists to avoid.
    """
    lines: list[str] = []
    for k, val in sorted(z.items(), key=lambda kv: -abs(kv[1] or 0)):
        if val is None:
            continue
        if abs(val) < 1.5:
            continue
        direction = "above" if val > 0 else "below"
        lines.append(
            f"{k}: {val:+.2f} SD {direction} the reference corpus."
        )

    if not lines:
        return ["No feature departs from the reference corpus by more "
                "than 1.5 SD. This is unremarkable text for this corpus."]

    lines.append("")
    lines.append(
        "These are deviations from a corpus you assembled, not evidence "
        "of authorship. A deviation is a reason to read the passage "
        "closely, or to ask the author about it. It is not a finding."
    )
    return lines
