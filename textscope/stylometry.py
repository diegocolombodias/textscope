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
from typing import Optional, Sequence

from .calibration import ReferenceStats

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
# Rewrite suggestions
#
# Plainer alternatives for the flagged lexicon. These are candidates, not
# mandates — a suggested swap that breaks the sentence's meaning or register
# should be ignored. The point is to give the reader something concrete to
# react to instead of a bare flag.
# ---------------------------------------------------------------------------

SYNONYMS_EN: dict[str, list[str]] = {
    "delve": ["explore", "look into"],
    "intricate": ["complex", "detailed"],
    "pivotal": ["key", "central"],
    "crucial": ["important", "essential"],
    "seamless": ["smooth", "consistent"],
    "robust": ["reliable", "solid"],
    "novel": ["new"],
    "comprehensive": ["thorough", "complete"],
    "nuanced": ["detailed", "subtle"],
    "underscore": ["highlight", "show"],
    "underscores": ["highlights", "shows"],
    "leverage": ["use"],
    "leveraging": ["using"],
    "landscape": ["field", "area"],
    "realm": ["field", "domain"],
    "tapestry": ["mix", "combination"],
    "furthermore": ["also", "and", "(often just cut it)"],
    "moreover": ["also", "and", "(often just cut it)"],
    "additionally": ["also", "and", "(often just cut it)"],
    "notably": ["in particular", "(often just cut it)"],
    "importantly": ["(often just cut it)"],
    "paramount": ["essential", "critical"],
    "multifaceted": ["varied", "complex"],
    "holistic": ["overall", "integrated"],
    "meticulous": ["careful"],
    "meticulously": ["carefully"],
    "testament": ["evidence", "sign"],
    "showcase": ["demonstrate", "show"],
    "showcases": ["demonstrates", "shows"],
    "foster": ["support", "encourage"],
    "fostering": ["supporting", "encouraging"],
    "encompass": ["include", "cover"],
    "encompasses": ["includes", "covers"],
    "extensive": ["wide", "broad"],
    "significant": ["important", "(or give the actual number)"],
    "vital": ["important", "necessary"],
}

SYNONYMS_PT: dict[str, list[str]] = {
    "crucial": ["importante", "essencial"],
    "fundamental": ["essencial", "central"],
    "robusto": ["confiável", "sólido"],
    "robusta": ["confiável", "sólida"],
    "abrangente": ["amplo", "completo"],
    "aprofundar": ["explorar em detalhe", "detalhar"],
    "aprofundado": ["detalhado"],
    "cenário": ["contexto", "situação"],
    "panorama": ["visão geral", "contexto"],
    "paradigma": ["modelo", "abordagem"],
    "ademais": ["além disso", "também"],
    "outrossim": ["também"],
    "ressaltar": ["destacar", "mostrar"],
    "ressalta": ["destaca", "mostra"],
    "destacar": ["mostrar", "apontar"],
    "destaca": ["mostra", "aponta"],
    "primordial": ["essencial", "principal"],
    "multifacetado": ["variado", "complexo"],
    "holístico": ["integrado", "geral"],
    "meticuloso": ["cuidadoso", "detalhado"],
    "notadamente": ["principalmente", "(muitas vezes dá para cortar)"],
    "sobretudo": ["principalmente"],
    "significativo": ["importante", "relevante"],
    "relevante": ["importante"],
    "inovador": ["novo", "original"],
}


def _doc_msg(lang: str, en: str, pt: str) -> str:
    return pt if lang == "pt" else en


def _lexicon_suggestion(hits: Sequence[str], lang: str) -> str:
    table = SYNONYMS_PT if lang == "pt" else SYNONYMS_EN
    parts = []
    for w in hits:
        alts = table.get(w)
        parts.append(f"'{w}' -> {' / '.join(alts)}" if alts else f"'{w}'")
    return _doc_msg(
        lang,
        en="Vocabulary over-represented in LLM output. Consider: "
           + "; ".join(parts) + ".",
        pt="Vocabulário superrepresentado em texto gerado por LLM. "
           "Considere: " + "; ".join(parts) + ".",
    )


def _scaffold_suggestion(lang: str) -> str:
    return _doc_msg(
        lang,
        en="Formulaic opener/scaffold phrase. Try cutting it or leading "
           "with the actual claim — it usually carries no information.",
        pt="Frase de abertura formulaica ('andaime'). Tente cortá-la ou "
           "entrar direto na afirmação — normalmente ela não carrega "
           "informação nova.",
    )


def _hedge_suggestion(n: int, lang: str) -> str:
    return _doc_msg(
        lang,
        en=f"{n} probability/frequency hedges stacked in one sentence. "
           "Keep at most one — stacked hedges weaken the claim and read "
           "as mechanical.",
        pt=f"{n} advérbios de probabilidade/frequência empilhados na "
           "mesma frase. Mantenha no máximo um — hedges empilhados "
           "enfraquecem a afirmação e soam mecânicos.",
    )


def _tricolon_suggestion(n: int, lang: str) -> str:
    return _doc_msg(
        lang,
        en="Triadic structure ('A, B, and C'). Fine on its own; if this "
           "pattern recurs across the document, vary the rhythm of one "
           "limb or split the sentence.",
        pt="Estrutura em tríade ('A, B e C'). Natural isoladamente; se o "
           "padrão se repetir no documento, varie o ritmo de uma das "
           "partes ou quebre a frase.",
    )


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
    suggestions: list[str] = field(default_factory=list)


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
    document_suggestions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "mean_sentence_length": round(self.mean_len, 2),
            "stdev_sentence_length": round(self.stdev_len, 2),
            "burstiness_cv": round(self.burstiness, 3),
            "tricolon_count": self.tricolon_count,
            "scaffold_phrases": self.scaffold_count,
            "lexicon_hits": self.lexicon_hits,
            "lexicon_hits_total": sum(self.lexicon_hits.values()),
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


def analyze_style(
    text: str, lang: str = "en", reference: Optional[ReferenceStats] = None
) -> StyleReport:
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
            f.suggestions.append(_tricolon_suggestion(tri, lang))

        for pat in scaffolds:
            if re.search(pat, s, flags=re.IGNORECASE):
                scaffold_total += 1
                f.flags.append("scaffold phrase")
                f.suggestions.append(_scaffold_suggestion(lang))
                break

        hits = sorted(set(w for w in sw if w in lexicon))
        if hits:
            f.flags.append("lexicon: " + ", ".join(hits[:4]))
            f.suggestions.append(_lexicon_suggestion(hits[:4], lang))

        n_hedge = sum(1 for w in sw if w in hedges)
        if n_hedge >= 2:
            hedge_stack_total += 1
            f.flags.append(f"stacked hedges x{n_hedge}")
            f.suggestions.append(_hedge_suggestion(n_hedge, lang))

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

    lexicon_hits = dict(sorted(lexicon_hits.items(), key=lambda kv: -kv[1]))
    em_dash_per_1k = 1000.0 * em_dashes / n_words

    doc_suggestions = _document_suggestions(
        lang=lang,
        burstiness=burstiness,
        n_sentences=len(lengths),
        em_dash_per_1k=em_dash_per_1k,
        n_words=n_words,
        opener_rep=opener_rep,
        n_paragraphs=len(openers),
        tricolon_total=tricolon_total,
        scaffold_total=scaffold_total,
        lexicon_hits=lexicon_hits,
        hedge_stack_total=hedge_stack_total,
        reference=reference,
    )

    return StyleReport(
        sentences=feats,
        mean_len=mean_len,
        stdev_len=stdev_len,
        burstiness=burstiness,
        tricolon_count=tricolon_total,
        scaffold_count=scaffold_total,
        lexicon_hits=lexicon_hits,
        hedge_stack_count=hedge_stack_total,
        em_dash_per_1k=em_dash_per_1k,
        opener_repetition=opener_rep,
        document_suggestions=doc_suggestions,
    )


def _z(key: str, value: float, reference: Optional[ReferenceStats]):
    """z-score of `value` against reference.feature_{means,stdevs}[key].

    Returns (z, mean) when the reference has usable stats for this key,
    else None — callers fall back to the fixed threshold in that case
    (no reference loaded, or an older reference.json missing this field).
    """
    if reference is None:
        return None
    mean = reference.feature_means.get(key)
    sd = reference.feature_stdevs.get(key)
    if mean is None or not sd:
        return None
    return (value - mean) / sd, mean


# z beyond this (either direction, as relevant) counts as "notably
# different from the reference corpus" — matches calibration.interpret()'s
# own cutoff, so the two don't disagree about what's worth mentioning.
_Z_CUTOFF = 1.5


def _document_suggestions(
    *,
    lang: str,
    burstiness: float,
    n_sentences: int,
    em_dash_per_1k: float,
    n_words: int,
    opener_rep: float,
    n_paragraphs: int,
    tricolon_total: int,
    scaffold_total: int,
    lexicon_hits: dict[str, int],
    hedge_stack_total: int,
    reference: Optional[ReferenceStats] = None,
) -> list[str]:
    """Document-wide observations, each paired with a concrete action.

    Without a reference corpus, thresholds are heuristic, tuned to flag
    only patterns clearly outside ordinary prose variation. With one
    (`textscope calibrate` on a corpus of real papers — see
    reference_corpus/ and README.md), each check instead compares against
    that corpus by z-score and speaks in its terms: "is this unusual for
    the kind of writing you calibrated against", not "does it cross a
    number picked in the abstract".

    Split into two groups because rewrite.py auto-applies fixes for the
    mechanical ones (scaffold phrases, flagged vocabulary, stacked
    hedges) but leaves the structural ones for a human — there is no
    single correct way to rebalance sentence rhythm or paragraph openers.
    """
    return structural_document_suggestions(
        lang=lang, burstiness=burstiness, n_sentences=n_sentences,
        em_dash_per_1k=em_dash_per_1k, n_words=n_words,
        opener_rep=opener_rep, n_paragraphs=n_paragraphs,
        tricolon_total=tricolon_total, reference=reference,
    ) + mechanical_document_suggestions(
        lang=lang, scaffold_total=scaffold_total, lexicon_hits=lexicon_hits,
        hedge_stack_total=hedge_stack_total, reference=reference,
    )


def structural_document_suggestions(
    *,
    lang: str,
    burstiness: float,
    n_sentences: int,
    em_dash_per_1k: float,
    n_words: int,
    opener_rep: float,
    n_paragraphs: int,
    tricolon_total: int = 0,
    reference: Optional[ReferenceStats] = None,
) -> list[str]:
    """Document notes that rewrite.py never auto-applies."""
    out: list[str] = []

    if n_sentences >= 5:
        zr = _z("burstiness_cv", burstiness, reference)
        if zr is not None:
            z, mean = zr
            if z <= -_Z_CUTOFF:
                out.append(_doc_msg(
                    lang,
                    en=f"Sentence-length variety (burstiness CV "
                       f"{burstiness:.2f}) is {abs(z):.1f} SD below your "
                       f"reference corpus, where papers average "
                       f"{mean:.2f}. Human academic prose usually mixes "
                       "short and long sentences more than this; try "
                       "splitting a few sentences apart or combining "
                       "others.",
                    pt=f"A variedade no comprimento das frases (CV "
                       f"{burstiness:.2f}) está {abs(z):.1f} DP abaixo do "
                       f"seu corpus de referência, onde a média é "
                       f"{mean:.2f}. Prosa acadêmica humana costuma "
                       "alternar frases curtas e longas mais do que "
                       "isso; tente quebrar algumas frases ou combinar "
                       "outras.",
                ))
        elif burstiness < 0.35:
            out.append(_doc_msg(
                lang,
                en=f"Sentence length is unusually uniform (coefficient "
                   f"of variation {burstiness:.2f}). Human prose tends "
                   "to mix short and long sentences; try deliberately "
                   "splitting a few sentences apart or combining others.",
                pt=f"O comprimento das frases é incomumente uniforme "
                   f"(coeficiente de variação {burstiness:.2f}). Prosa "
                   "humana costuma alternar frases curtas e longas; "
                   "tente quebrar algumas frases ou combinar outras "
                   "deliberadamente.",
            ))

    if n_words >= 60:
        zr = _z("em_dashes_per_1k_words", em_dash_per_1k, reference)
        if zr is not None:
            z, mean = zr
            if z >= _Z_CUTOFF:
                out.append(_doc_msg(
                    lang,
                    en=f"Em dashes appear at {em_dash_per_1k:.1f} per "
                       f"1,000 words, {z:.1f} SD above your reference "
                       f"corpus (average {mean:.1f}/1k there). Replace "
                       "some with commas, parentheses, or a full stop.",
                    pt=f"Travessões aparecem a {em_dash_per_1k:.1f} por "
                       f"1.000 palavras, {z:.1f} DP acima do seu corpus "
                       f"de referência (média de {mean:.1f}/mil lá). "
                       "Substitua parte deles por vírgulas, parênteses "
                       "ou ponto final.",
                ))
        elif em_dash_per_1k > 8.0:
            out.append(_doc_msg(
                lang,
                en=f"Em dashes appear at {em_dash_per_1k:.1f} per 1,000 "
                   "words, well above typical prose. Replace some with "
                   "commas, parentheses, or a full stop.",
                pt=f"Travessões aparecem a {em_dash_per_1k:.1f} por "
                   "1.000 palavras, acima do comum. Substitua parte "
                   "deles por vírgulas, parênteses ou ponto final.",
            ))

    if n_paragraphs >= 3:
        zr = _z("paragraph_opener_repetition", opener_rep, reference)
        if zr is not None:
            z, mean = zr
            if z >= _Z_CUTOFF:
                out.append(_doc_msg(
                    lang,
                    en=f"{opener_rep:.0%} of paragraphs open with the "
                       f"same word, {z:.1f} SD above your reference "
                       f"corpus (average {mean:.0%} there). Vary "
                       "paragraph openers to avoid a templated feel.",
                    pt=f"{opener_rep:.0%} dos parágrafos começam com a "
                       f"mesma palavra, {z:.1f} DP acima do seu corpus "
                       f"de referência (média de {mean:.0%} lá). Varie "
                       "as aberturas de parágrafo para evitar um efeito "
                       "de modelo repetido.",
                ))
        elif opener_rep > 0.34:
            out.append(_doc_msg(
                lang,
                en=f"{opener_rep:.0%} of paragraphs open with the same "
                   "word. Vary paragraph openers to avoid a templated "
                   "feel.",
                pt=f"{opener_rep:.0%} dos parágrafos começam com a "
                   "mesma palavra. Varie as aberturas de parágrafo para "
                   "evitar um efeito de modelo repetido.",
            ))

    zr = _z("tricolon_count", tricolon_total, reference)
    if zr is not None:
        z, mean = zr
        if z >= _Z_CUTOFF:
            out.append(_doc_msg(
                lang,
                en=f"{tricolon_total} triadic ('A, B, and C') "
                   f"structures, {z:.1f} SD above your reference corpus "
                   f"(average {mean:.1f} there). Fine in small doses; "
                   "vary the rhythm of one limb in a few of them.",
                pt=f"{tricolon_total} estruturas em tríade ('A, B e C'), "
                   f"{z:.1f} DP acima do seu corpus de referência (média "
                   f"de {mean:.1f} lá). São naturais em doses pequenas; "
                   "varie o ritmo de um dos elementos em algumas delas.",
            ))
    elif tricolon_total >= 1 and reference is None:
        out.append(_doc_msg(
            lang,
            en=f"{tricolon_total} triadic ('A, B, and C') structure(s) "
               "in the document. Not a problem by itself — worth a look "
               "only if it recurs on rereading.",
            pt=f"{tricolon_total} estrutura(s) em tríade ('A, B e C') no "
               "documento. Não é um problema isolado — vale olhar só se "
               "se repetir numa releitura.",
        ))

    return out


def mechanical_document_suggestions(
    *,
    lang: str,
    scaffold_total: int,
    lexicon_hits: dict[str, int],
    hedge_stack_total: int = 0,
    reference: Optional[ReferenceStats] = None,
) -> list[str]:
    """Document notes for issues rewrite.py auto-fixes.

    Kept separate from structural_document_suggestions so rewrite.py's
    "still needs a human look" list doesn't re-list what it just fixed.
    """
    out: list[str] = []

    zr = _z("scaffold_phrases", scaffold_total, reference)
    if zr is not None:
        z, mean = zr
        if z >= _Z_CUTOFF:
            out.append(_doc_msg(
                lang,
                en=f"{scaffold_total} formulaic scaffold phrases, "
                   f"{z:.1f} SD above your reference corpus (average "
                   f"{mean:.1f} there). The rewrite below already drops "
                   "the ones safe to cut.",
                pt=f"{scaffold_total} frases de andaime formulaicas, "
                   f"{z:.1f} DP acima do seu corpus de referência (média "
                   f"de {mean:.1f} lá). A reescrita abaixo já remove as "
                   "que são seguras de cortar.",
            ))
    elif scaffold_total >= 2:
        out.append(_doc_msg(
            lang,
            en=f"{scaffold_total} formulaic scaffold phrases found "
               "across the document (see per-sentence notes). Cutting "
               "most of them tightens the prose without losing meaning.",
            pt=f"{scaffold_total} frases de andaime formulaicas "
               "encontradas no documento (veja as notas por frase). "
               "Cortar a maioria delas deixa o texto mais direto sem "
               "perder sentido.",
        ))

    total_lexicon = sum(lexicon_hits.values())
    top = ", ".join(f"{w} ({c}x)" for w, c in list(lexicon_hits.items())[:5])
    zr = _z("lexicon_hits_total", total_lexicon, reference)
    if zr is not None:
        z, mean = zr
        if z >= _Z_CUTOFF:
            out.append(_doc_msg(
                lang,
                en=f"Vocabulary flagged as LLM-frequent appears "
                   f"{total_lexicon} times, {z:.1f} SD above your "
                   f"reference corpus (average {mean:.1f} there). Most "
                   f"common: {top}.",
                pt=f"Vocabulário sinalizado como frequente em LLM "
                   f"aparece {total_lexicon} vezes, {z:.1f} DP acima do "
                   f"seu corpus de referência (média de {mean:.1f} lá). "
                   f"Mais comuns: {top}.",
            ))
    elif total_lexicon >= 5:
        out.append(_doc_msg(
            lang,
            en=f"Vocabulary flagged as LLM-frequent appears "
               f"{total_lexicon} times. Most common: {top}. A pass "
               "replacing repeated instances with plainer synonyms "
               "usually helps.",
            pt=f"Vocabulário sinalizado como frequente em LLM aparece "
               f"{total_lexicon} vezes. Mais comuns: {top}. Uma revisão "
               "trocando as repetições por sinônimos mais simples "
               "costuma ajudar.",
        ))

    zr = _z("stacked_hedges", hedge_stack_total, reference)
    if zr is not None:
        z, mean = zr
        if z >= _Z_CUTOFF:
            out.append(_doc_msg(
                lang,
                en=f"{hedge_stack_total} sentences stack 2+ hedge words "
                   f"(typically/often/possibly...), {z:.1f} SD above "
                   f"your reference corpus (average {mean:.1f} there). "
                   "The rewrite below already collapses these to one.",
                pt=f"{hedge_stack_total} frases empilham 2+ hedges "
                   f"(tipicamente/frequentemente/possivelmente...), "
                   f"{z:.1f} DP acima do seu corpus de referência (média "
                   f"de {mean:.1f} lá). A reescrita abaixo já reduz isso "
                   "a um só.",
            ))
    elif hedge_stack_total >= 2:
        out.append(_doc_msg(
            lang,
            en=f"{hedge_stack_total} sentences stack 2+ hedge words "
               "(typically/often/possibly...) across the document. The "
               "rewrite below already collapses these to one.",
            pt=f"{hedge_stack_total} frases empilham 2+ hedges "
               "(tipicamente/frequentemente/possivelmente...) no "
               "documento. A reescrita abaixo já reduz isso a um só.",
        ))

    return out
