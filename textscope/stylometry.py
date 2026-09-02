"""
Stylometric markers associated with LLM-generated prose.

None of these features is evidence on its own. They are descriptive
statistics that make editorial patterns visible. Interpretation is the
user's job, not this module's.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Sequence

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

# Terms over-represented in LLM output relative to human academic prose.
# Frequencies are corpus-dependent; treat these as candidates to inspect,
# not as a blacklist.
LLM_FREQUENT_EN = {
    "delve", "intricate", "pivotal", "crucial", "seamless", "robust",
    "novel", "comprehensive", "nuanced", "underscore", "underscores",
    "leverage", "leveraging", "landscape", "realm", "tapestry",
    "furthermore", "moreover", "additionally", "notably", "importantly",
    "paramount", "multifaceted", "holistic", "meticulous", "meticulously",
    "testament", "showcase", "showcases", "foster", "fostering",
    "encompass", "encompasses", "extensive", "significant", "vital",
}

LLM_FREQUENT_PT = {
    "crucial", "fundamental", "robusto", "robusta", "abrangente",
    "aprofundar", "aprofundado", "cenário", "panorama", "paradigma",
    "ademais", "outrossim", "ressaltar", "ressalta", "destacar",
    "destaca", "primordial", "multifacetado", "holístico", "meticuloso",
    "notadamente", "sobretudo", "significativo", "relevante", "inovador",
}

HEDGE_STACK_EN = {"typically", "frequently", "generally", "often",
                  "usually", "commonly", "potentially", "possibly"}
HEDGE_STACK_PT = {"tipicamente", "frequentemente", "geralmente",
                  "normalmente", "possivelmente", "potencialmente"}

# Formulaic scaffolding phrases.
SCAFFOLD_EN = [
    r"\bin this (article|paper|study|work),?\s",
    r"\bit is (important|worth|crucial) to (note|mention|highlight)\b",
    r"\bin (the )?(first|second|third) (component|part|section|place)\b",
    r"\bfurthermore\b", r"\bmoreover\b",
    r"\boverall,\s", r"\bin conclusion,\s",
    r"\bplays? a (crucial|vital|key|pivotal) role\b",
    r"\bnot only .{1,60}? but also\b",
]
SCAFFOLD_PT = [
    r"\bneste (artigo|trabalho|estudo),?\s",
    r"\bé (importante|fundamental|crucial) (notar|destacar|ressaltar)\b",
    r"\b(no|na) (primeiro|segundo|terceiro) (componente|parte|seção)\b",
    r"\balém disso\b", r"\bademais\b", r"\bpor outro lado\b",
    r"\bem suma,\s", r"\bem conclusão,\s",
    r"\bdesempenha um papel (crucial|fundamental|central)\b",
    r"\bnão apenas .{1,60}? mas também\b",
]

ABBREV = {"e.g", "i.e", "cf", "et al", "vs", "Dr", "Prof", "Fig", "Eq",
          "approx", "resp", "ca", "no", "cap", "ed", "org", "p", "pp"}


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Sentence splitter tolerant of academic abbreviations and math."""
    # Protect inline math so '$M = 10.000$' is not split at the period.
    math_spans: list[str] = []

    def _stash(m: re.Match) -> str:
        math_spans.append(m.group(0))
        return f"\x00MATH{len(math_spans) - 1}\x00"

    protected = re.sub(r"\$[^$]*\$", _stash, text)

    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Þ(\\])", protected)

    merged: list[str] = []
    for part in parts:
        tail = part.rstrip()[:-1].split()[-1] if part.rstrip()[:-1].split() else ""
        if merged and tail.rstrip(".") in ABBREV:
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)

    out = []
    for s in merged:
        for i, span in enumerate(math_spans):
            s = s.replace(f"\x00MATH{i}\x00", span)
        s = s.strip()
        if s:
            out.append(s)
    return out


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]*", text.lower())


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

@dataclass
class SentenceFeatures:
    index: int
    text: str
    n_words: int
    flags: list[str] = field(default_factory=list)


@dataclass
class StyleReport:
    sentences: list[SentenceFeatures]
    mean_len: float
    stdev_len: float
    burstiness: float          # coefficient of variation of sentence length
    tricolon_count: int
    scaffold_count: int
    lexicon_hits: dict[str, int]
    hedge_stack_count: int
    em_dash_per_1k: float
    opener_repetition: float   # share of paragraphs opening the same way

    def as_dict(self) -> dict:
        return {
            "mean_sentence_length": round(self.mean_len, 2),
            "stdev_sentence_length": round(self.stdev_len, 2),
            "burstiness_cv": round(self.burstiness, 3),
            "tricolon_count": self.tricolon_count,
            "scaffold_phrases": self.scaffold_count,
            "lexicon_hits": self.lexicon_hits,
            "stacked_hedges": self.hedge_stack_count,
            "em_dashes_per_1k_words": round(self.em_dash_per_1k, 2),
            "paragraph_opener_repetition": round(self.opener_repetition, 3),
        }


def _count_tricolons(sentence: str) -> int:
    """Detect 'A, B, and C' / 'A, B e C' triples of comparable length."""
    patterns = [
        r"([^,;:]{4,60}),\s+([^,;:]{4,60}),\s+and\s+([^,;.:]{4,60})",
        r"([^,;:]{4,60}),\s+([^,;:]{4,60}),?\s+e\s+([^,;.:]{4,60})",
    ]
    total = 0
    for pat in patterns:
        for m in re.finditer(pat, sentence, flags=re.IGNORECASE):
            lengths = [len(g.split()) for g in m.groups()]
            # Parallelism: the three limbs are of similar weight.
            if max(lengths) <= 3 * min(lengths):
                total += 1
    return total


def analyze_style(text: str, lang: str = "en") -> StyleReport:
    lexicon = LLM_FREQUENT_PT if lang == "pt" else LLM_FREQUENT_EN
    hedges = HEDGE_STACK_PT if lang == "pt" else HEDGE_STACK_EN
    scaffolds = SCAFFOLD_PT if lang == "pt" else SCAFFOLD_EN

    sentences = split_sentences(text)
    words_all = tokenize_words(text)

    feats: list[SentenceFeatures] = []
    tricolon_total = 0
    scaffold_total = 0
    hedge_stack_total = 0

    for i, s in enumerate(sentences):
        sw = tokenize_words(s)
        f = SentenceFeatures(index=i, text=s, n_words=len(sw))

        tri = _count_tricolons(s)
        if tri:
            tricolon_total += tri
            f.flags.append(f"tricolon x{tri}")

        for pat in scaffolds:
            if re.search(pat, s, flags=re.IGNORECASE):
                scaffold_total += 1
                f.flags.append("scaffold phrase")
                break

        hits = [w for w in sw if w in lexicon]
        if hits:
            f.flags.append("lexicon: " + ", ".join(sorted(set(hits))[:4]))

        n_hedge = sum(1 for w in sw if w in hedges)
        if n_hedge >= 2:
            hedge_stack_total += 1
            f.flags.append(f"stacked hedges x{n_hedge}")

        feats.append(f)

    lengths = [f.n_words for f in feats] or [0]
    mean_len = statistics.fmean(lengths)
    stdev_len = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
    burstiness = stdev_len / mean_len if mean_len else 0.0

    lexicon_hits: dict[str, int] = {}
    for w in words_all:
        if w in lexicon:
            lexicon_hits[w] = lexicon_hits.get(w, 0) + 1

    n_words = max(len(words_all), 1)
    em_dashes = text.count("—") + text.count("---")

    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    openers = [tokenize_words(p)[:2] for p in paragraphs if tokenize_words(p)]
    opener_rep = 0.0
    if len(openers) > 1:
        first_words = [o[0] for o in openers if o]
        most = max(set(first_words), key=first_words.count)
        opener_rep = first_words.count(most) / len(first_words)

    return StyleReport(
        sentences=feats,
        mean_len=mean_len,
        stdev_len=stdev_len,
        burstiness=burstiness,
        tricolon_count=tricolon_total,
        scaffold_count=scaffold_total,
        lexicon_hits=dict(sorted(lexicon_hits.items(),
                                 key=lambda kv: -kv[1])),
        hedge_stack_count=hedge_stack_total,
        em_dash_per_1k=1000.0 * em_dashes / n_words,
        opener_repetition=opener_rep,
    )
