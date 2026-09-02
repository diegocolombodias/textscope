"""
Deterministic, rule-based auto-rewrite of the mechanical stylometric flags.

This applies only edits with a single unambiguous outcome:

  * dropping a formulaic scaffold clause ("In this article, we...")
  * swapping a flagged word for the plainer alternative listed in
    stylometry.SYNONYMS_EN / SYNONYMS_PT
  * collapsing a contiguous run of stacked hedges ("typically, frequently,
    and generally") down to the first one

It never touches sentence structure, paragraph order, or anything that is
a matter of judgment — tricolon phrasing, sentence-length variety, em dash
rate, paragraph openers. Those come back in `unresolved` instead, because
there is no single correct rewrite for them.

This is NOT a paraphraser and does not call any model or generate new
text. Every change is one of the three mechanical rules above and is
listed, per sentence, in SentenceRewrite.edits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .calibration import ReferenceStats
from .stylometry import (
    HEDGE_STACK_EN,
    HEDGE_STACK_PT,
    LLM_FREQUENT_EN,
    LLM_FREQUENT_PT,
    SYNONYMS_EN,
    SYNONYMS_PT,
    _count_tricolons,
    _doc_msg,
    analyze_style,
    split_sentences,
    structural_document_suggestions,
    tokenize_words,
)

_CONTINUATION_STRIP = {
    "en": re.compile(r"^(?:that|which)\b\s*", re.IGNORECASE),
    "pt": re.compile(r"^que\b\s*", re.IGNORECASE),
}

_MIN_SCAFFOLD_REMAINDER = 8  # chars; below this, a scaffold strip is unsafe

# Only clauses that are pure preamble/connector — deleting them can never
# strand a verb or object. stylometry.SCAFFOLD_EN/PT (used for flagging)
# also includes role clichés ("plays a crucial role in...") and the
# correlative "not only... but also", which are load-bearing parts of the
# sentence's grammar; those stay flagged but are never auto-deleted here.
_DROPPABLE_SCAFFOLD_EN = [
    r"\bin this (article|paper|study|work),?\s",
    r"\bit is (important|worth|crucial) to (note|mention|highlight)\b",
    r"\bfurthermore\b",
    r"\bmoreover\b",
    r"\boverall,\s",
    r"\bin conclusion,\s",
]
_DROPPABLE_SCAFFOLD_PT = [
    r"\bneste (artigo|trabalho|estudo),?\s",
    r"\bé (importante|fundamental|crucial) (notar|destacar|ressaltar)\b",
    r"\balém disso\b",
    r"\bademais\b",
    r"\bpor outro lado\b",
    r"\bem suma,\s",
    r"\bem conclusão,\s",
]


@dataclass
class SentenceRewrite:
    index: int
    original: str
    rewritten: str
    edits: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "original": self.original,
            "rewritten": self.rewritten,
            "changed": self.rewritten != self.original,
            "edits": self.edits,
        }


@dataclass
class RewriteResult:
    text: str
    sentences: list[SentenceRewrite]
    unresolved: list[str]

    def changed_count(self) -> int:
        return sum(1 for s in self.sentences if s.rewritten != s.original)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "changed_sentences": self.changed_count(),
            "total_sentences": len(self.sentences),
            "sentences": [s.as_dict() for s in self.sentences],
            "unresolved": self.unresolved,
        }


def _strip_scaffold(sentence: str, patterns: list[str], lang: str) -> tuple[str, bool]:
    """Drop every formulaic scaffold clause, not just the first.

    A sentence can stack two ("Neste artigo, é importante destacar que...")
    so this rescans the patterns against the shrinking remainder until
    none match, up to a small round limit.
    """
    remainder = sentence
    changed_any = False

    for _ in range(3):
        stripped_this_round = False
        for pat in patterns:
            m = re.search(pat, remainder, flags=re.IGNORECASE)
            if not m:
                continue
            candidate = remainder[: m.start()] + remainder[m.end() :]
            candidate = re.sub(r"^[,\s]+", "", candidate)
            candidate = _CONTINUATION_STRIP[lang].sub("", candidate, count=1)
            candidate = candidate.strip()
            if len(candidate) < _MIN_SCAFFOLD_REMAINDER:
                continue  # too little left — leave this clause alone
            remainder = candidate
            changed_any = True
            stripped_this_round = True
            break  # restart the pattern scan against the new remainder
        if not stripped_this_round:
            break

    if not changed_any:
        return sentence, False
    return remainder[0].upper() + remainder[1:], True


def _swap_lexicon(
    sentence: str, present: set[str], synonyms: dict[str, list[str]], lang: str
) -> tuple[str, list[str]]:
    """Replace each flagged word with its listed alternative.

    For English, also re-agrees a directly preceding indefinite article
    ("a crucial" -> "an important"), since a word swap can silently break
    a/an agreement. The article is only touched when it sits immediately
    before the swapped word, so unrelated "a"/"an" elsewhere in the
    sentence (e.g. "an hour") are never rescanned.
    """
    candidates = sorted((w for w in present if w in synonyms), key=len, reverse=True)
    if not candidates:
        return sentence, []

    swapped: list[str] = []
    word_alt = "|".join(re.escape(w) for w in candidates)
    if lang == "en":
        pattern = re.compile(rf"(?:\b([Aa]n?)\s+)?\b(?P<word>{word_alt})\b",
                              re.IGNORECASE)
    else:
        pattern = re.compile(rf"\b(?P<word>{word_alt})\b", re.IGNORECASE)

    def _repl(m: re.Match) -> str:
        word = m.group("word")
        article = m.group(1) if lang == "en" else None
        repl = synonyms[word.lower()][0]
        if word[:1].isupper():
            repl = repl[:1].upper() + repl[1:]
        swapped.append(f"{word} -> {repl}")
        if article:
            art = "an" if repl[:1].lower() in "aeiou" else "a"
            if article[:1].isupper():
                art = art.capitalize()
            return f"{art} {repl}"
        return repl

    return pattern.sub(_repl, sentence), swapped


_PHRASE_SWAPS_EN: list[tuple[re.Pattern, str]] = [
    # "delve" is a synonym fallback for the bare verb, but "delve into" is
    # the overwhelmingly common form and needs the preposition dropped
    # too, or the swap reads as "explore into X".
    (re.compile(r"\bdelve into\b", re.IGNORECASE), "explore"),
]
_PHRASE_SWAPS_PT: list[tuple[re.Pattern, str]] = []


def _swap_phrases(sentence: str, lang: str) -> tuple[str, list[str]]:
    phrases = _PHRASE_SWAPS_PT if lang == "pt" else _PHRASE_SWAPS_EN
    edits: list[str] = []

    def _make_repl(replacement: str):
        def _repl(m: re.Match) -> str:
            matched = m.group(0)
            out = replacement
            if matched[:1].isupper():
                out = out[:1].upper() + out[1:]
            edits.append(f"{matched} -> {out}")
            return out
        return _repl

    for pat, repl in phrases:
        sentence = pat.sub(_make_repl(repl), sentence)
    return sentence, edits


def _collapse_hedges(sentence: str, hedges: set[str]) -> tuple[str, bool]:
    words = sorted(hedges, key=len, reverse=True)
    alt = "|".join(re.escape(w) for w in words)
    # Handles a plain comma, an Oxford comma before "and"/"e", a bare
    # "and"/"e", or plain whitespace between consecutive hedges.
    connector = r"(?:\s*,\s*(?:e\s+|and\s+)?|\s+e\s+|\s+and\s+|\s+)"
    single = re.compile(rf"\b(?:{alt})\b", re.IGNORECASE)
    run = re.compile(
        rf"\b(?:{alt})\b(?:{connector}\b(?:{alt})\b)+",
        re.IGNORECASE,
    )

    changed = False

    def _repl(m: re.Match) -> str:
        nonlocal changed
        changed = True
        first = single.search(m.group(0))
        return first.group(0) if first else m.group(0)

    return run.sub(_repl, sentence), changed


def rewrite_text(
    text: str, lang: str = "en", reference: Optional[ReferenceStats] = None
) -> RewriteResult:
    lexicon = LLM_FREQUENT_PT if lang == "pt" else LLM_FREQUENT_EN
    synonyms = SYNONYMS_PT if lang == "pt" else SYNONYMS_EN
    hedges = HEDGE_STACK_PT if lang == "pt" else HEDGE_STACK_EN
    scaffolds = _DROPPABLE_SCAFFOLD_PT if lang == "pt" else _DROPPABLE_SCAFFOLD_EN

    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    out_paragraphs: list[str] = []
    sentence_rewrites: list[SentenceRewrite] = []
    idx = 0
    tricolon_total = 0

    for para in paragraphs:
        rewritten_here: list[str] = []
        for s in split_sentences(para):
            edits: list[str] = []
            cur = s

            tricolon_total += _count_tricolons(s)

            cur, did_strip = _strip_scaffold(cur, scaffolds, lang)
            if did_strip:
                edits.append(_doc_msg(
                    lang,
                    en="dropped scaffold phrase",
                    pt="frase de andaime removida",
                ))

            cur, hedge_changed = _collapse_hedges(cur, hedges)
            if hedge_changed:
                edits.append(_doc_msg(
                    lang,
                    en="collapsed stacked hedges to one",
                    pt="hedges empilhados reduzidos a um",
                ))

            cur, phrase_swaps = _swap_phrases(cur, lang)
            edits.extend(phrase_swaps)

            present = set(tokenize_words(cur)) & lexicon
            cur, swaps = _swap_lexicon(cur, present, synonyms, lang)
            edits.extend(swaps)

            sentence_rewrites.append(SentenceRewrite(idx, s, cur, edits))
            rewritten_here.append(cur)
            idx += 1
        out_paragraphs.append(" ".join(rewritten_here))

    style = analyze_style(text, lang=lang)
    unresolved = structural_document_suggestions(
        lang=lang,
        burstiness=style.burstiness,
        n_sentences=len(style.sentences),
        em_dash_per_1k=style.em_dash_per_1k,
        n_words=max(sum(f.n_words for f in style.sentences), 1),
        opener_rep=style.opener_repetition,
        n_paragraphs=len(paragraphs),
        tricolon_total=tricolon_total,
        reference=reference,
    )

    return RewriteResult(
        text="\n\n".join(out_paragraphs),
        sentences=sentence_rewrites,
        unresolved=unresolved,
    )


def render_rewrite_text(result: RewriteResult) -> str:
    lines: list[str] = []
    add = lines.append

    add("=" * 72)
    add("SUGGESTED REWRITE (mechanical fixes applied automatically)")
    add("=" * 72)
    add("")
    add(f"{result.changed_count()} of {len(result.sentences)} sentences edited.")
    add("")
    add(result.text)

    if result.unresolved:
        add("")
        add("STILL NEEDS A HUMAN LOOK (not auto-applied — a judgment call):")
        for i, u in enumerate(result.unresolved, 1):
            add(f"  {i}. {u}")

    add("")
    add("This is a mechanical, rule-based edit — not a paraphrase and not "
        "a claim that the result reads better. Read it before keeping it.")
    return "\n".join(lines)
