"""Text + skill-evidence helpers.

Ported and adapted from verdix's resume-matcher ``_skill_evidenced`` / alias
logic: different wording should still count, and we match against a curated
vocabulary rather than every token. Pure-Python, no dependencies, no network.

Performance: every phrase pattern is compiled ONCE and cached. Whole-token
matching uses these cached patterns, so scoring the full 100K pool stays well
inside the CPU budget (the naive recompile-per-call version did not).
"""

from __future__ import annotations

import re

# Alias map: canonical skill -> phrases that imply it. Mirrors the spirit of
# verdix's SKILL_ALIASES, extended for the AI/IR vocabulary this JD cares about.
SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "embeddings": ("embedding", "sentence-transformers", "sentence transformers",
                   "bge", "e5", "word2vec", "vector representation"),
    "retrieval": ("retriever", "rag", "retrieval augmented", "dense retrieval",
                  "information retrieval", "ir system"),
    "information retrieval": ("ir", "retrieval", "search relevance"),
    "ranking": ("ranker", "rank", "learning to rank", "ltr", "re-rank", "rerank"),
    "learning to rank": ("ltr", "learning-to-rank", "lambdamart", "ranknet"),
    "recommendation systems": ("recommender", "recommendation", "recsys",
                               "collaborative filtering", "matrix factorization"),
    "vector search": ("ann", "approximate nearest neighbor", "nearest neighbor",
                       "knn search", "vector index"),
    "semantic search": ("semantic", "dense search", "neural search"),
    "hybrid search": ("hybrid retrieval", "bm25 + dense", "lexical + semantic"),
    "elasticsearch": ("elastic", "es cluster", "lucene"),
    "opensearch": ("open search",),
    "ndcg": ("normalized discounted cumulative gain",),
    "mrr": ("mean reciprocal rank",),
    "map": ("mean average precision",),
    "a/b testing": ("a/b test", "ab test", "split test", "online experiment"),
    "nlp": ("natural language processing", "text classification",
            "named entity", "language model"),
    "python": ("pandas", "numpy", "pytorch", "fastapi", "django", "flask"),
    "fine-tuning llms": ("fine-tune", "fine tuning", "finetuning", "lora", "qlora",
                         "peft", "instruction tuning"),
    "transformers": ("bert", "roberta", "gpt", "llm", "huggingface"),
}

_WORD_BOUND = r"(?<![a-z0-9])"
_WORD_BOUND_END = r"(?![a-z0-9])"

# Compile-once cache of whole-token patterns, keyed by the literal phrase.
_PATTERN_CACHE: dict[str, re.Pattern] = {}


def _pattern(phrase: str) -> re.Pattern:
    pat = _PATTERN_CACHE.get(phrase)
    if pat is None:
        pat = re.compile(_WORD_BOUND + re.escape(phrase.lower()) + _WORD_BOUND_END)
        _PATTERN_CACHE[phrase] = pat
    return pat


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def contains_phrase(text_lower: str, phrase: str) -> bool:
    """Whole-token containment so 'map' doesn't match 'mapping'.

    Fast pre-filter: a plain substring check (cheap) gates the regex (costly).
    If the literal substring isn't present, the whole-token match cannot be
    either, so we skip the regex entirely — the common case for a sparse vocab.
    """
    if not phrase:
        return False
    if phrase not in text_lower:
        return False
    return _pattern(phrase).search(text_lower) is not None


def evidences(text_lower: str, canonical: str) -> bool:
    """True if text evidences a canonical skill directly or via a known alias."""
    if contains_phrase(text_lower, canonical):
        return True
    return any(contains_phrase(text_lower, alias)
               for alias in SKILL_ALIASES.get(canonical, ()))


def any_phrase(text_lower: str, phrases) -> bool:
    return any(contains_phrase(text_lower, p) for p in phrases)


def count_phrases(text_lower: str, phrases) -> int:
    return sum(1 for p in phrases if contains_phrase(text_lower, p))


def build_canonical_matcher(canonicals) -> list[tuple[str, tuple[str, ...]]]:
    """Precompute, once, a (canonical, triggers) list for a set of canonical
    skills. ``triggers`` = the canonical itself plus its aliases. Used by the
    feature extractor to map a short skill NAME onto canonical rubric entries
    with cheap substring checks instead of an O(skills x vocab) regex loop.
    """
    out = []
    for c in canonicals:
        triggers = (c,) + SKILL_ALIASES.get(c, ())
        out.append((c, triggers))
    return out


def match_canonicals(name_lower: str,
                     matcher: list[tuple[str, tuple[str, ...]]]) -> list[str]:
    """Return canonicals whose name/alias appears as a whole token in ``name_lower``.

    ``name_lower`` is a short skill name, so the substring pre-filter in
    ``contains_phrase`` makes this effectively O(triggers) of cheap checks.
    """
    hits = []
    for canon, triggers in matcher:
        for t in triggers:
            if contains_phrase(name_lower, t):
                hits.append(canon)
                break
    return hits
