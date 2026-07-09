from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import unicodedata


TCM_TERMS = (
    "柴胡桂枝干姜汤",
    "心脾两虚",
    "肝阳上亢",
    "肝火上炎",
    "补益心脾",
    "归脾汤",
    "党参",
    "柴胡",
    "桂枝",
    "干姜",
    "失眠",
    "不寐",
    "证候",
    "中药",
    "方剂",
    "症状",
    "治法",
)

TERM_ALIASES = {
    "睡不着": ["失眠", "不寐"],
    "入睡困难": ["失眠", "不寐"],
    "夜寐不安": ["失眠", "不寐"],
    "偏头疼": ["头痛"],
}

ENTITY_TERMS = set(TCM_TERMS) | {alias for aliases in TERM_ALIASES.values() for alias in aliases}

STOP_WORDS = {
    "可以",
    "哪些",
    "什么",
    "怎么",
    "如何",
    "处理",
    "关联",
    "组成",
    "分析",
    "从",
    "吗",
    "呢",
    "的",
    "了",
    "和",
    "与",
    "及",
    "或",
    "是",
    "为",
}

STOP_CHARS = set("可以哪些什么怎么如何处理关联组成分析从吗呢的了和与及或是为")


@dataclass(frozen=True)
class QueryRewriteResult:
    answer_type_keywords: list[str] = field(default_factory=list)
    entities_from_query: list[str] = field(default_factory=list)


def tokenize_query(text: str, *, max_tokens: int = 256) -> list[str]:
    """RAGFlow-inspired lightweight Chinese query tokenization.

    The original RAGFlow implementation delegates to rag_tokenizer and term_weight.
    This local variant keeps the same retrieval-facing idea: normalize punctuation,
    preserve known domain terms, and add compact adjacent Chinese n-grams for fallback
    matching without introducing a heavyweight dependency.
    """

    normalized = _normalize(text)
    tokens: list[str] = []
    deferred_aliases: list[str] = []
    for raw_term, aliases in TERM_ALIASES.items():
        if raw_term in normalized:
            tokens.extend(aliases[:1])
            deferred_aliases.extend(aliases[1:])
    for term in sorted(TCM_TERMS, key=len, reverse=True):
        if term in normalized:
            tokens.append(term)
    tokens.extend(deferred_aliases)
    for segment in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9][a-z0-9_\-.+#]*", normalized):
        if segment in STOP_WORDS:
            continue
        if re.fullmatch(r"[a-z0-9][a-z0-9_\-.+#]*", segment):
            tokens.append(segment)
            continue
        if len(segment) <= 4:
            tokens.append(segment)
        for size in (2, 3, 4):
            if len(segment) < size:
                continue
            tokens.extend(segment[index : index + size] for index in range(len(segment) - size + 1))
    return _unique(token for token in tokens if token and token not in STOP_WORDS)[:max_tokens]


def weighted_token_dict(tokens: str | list[str]) -> dict[str, float]:
    if isinstance(tokens, str):
        token_list = tokens.split()
    else:
        token_list = tokens
    weights: dict[str, float] = {}
    for index, token in enumerate(token_list):
        if not token:
            continue
        weights[token] = weights.get(token, 0.0) + 0.4
        if index + 1 < len(token_list):
            bigram = token + token_list[index + 1]
            weights[bigram] = weights.get(bigram, 0.0) + 0.6
    return weights


def token_similarity(query_tokens: str | list[str], document_tokens: list[str | list[str]]) -> list[float]:
    query_weights = weighted_token_dict(query_tokens)
    return [_similarity(query_weights, weighted_token_dict(tokens)) for tokens in document_tokens]


def entity_keywords(keywords: list[str], *, max_entities: int = 5) -> list[str]:
    entities = [
        keyword
        for keyword in keywords
        if keyword in ENTITY_TERMS and keyword not in {"证候", "方剂", "症状", "治法", "中药"}
    ]
    return _unique(entities)[:max_entities]


def candidate_search_keywords(keywords: list[str], *, max_keywords: int = 8) -> list[str]:
    entities = entity_keywords(keywords, max_entities=max_keywords)
    if entities:
        return entities
    clean_keywords = [
        keyword
        for keyword in keywords
        if 2 <= len(keyword) <= 6 and keyword not in STOP_WORDS and not _looks_like_noise_ngram(keyword)
    ]
    prefixes = {
        keyword
        for keyword in clean_keywords
        for other in clean_keywords
        if keyword != other and other.startswith(keyword) and len(other) > len(keyword)
    }
    return _unique(keyword for keyword in clean_keywords if keyword not in prefixes)[:max_keywords]


def parse_query_rewrite(raw: str) -> QueryRewriteResult:
    data = _loads_jsonish(raw)
    return QueryRewriteResult(
        answer_type_keywords=_string_list(data.get("answer_type_keywords", [])),
        entities_from_query=_string_list(data.get("entities_from_query", []))[:5],
    )


def build_content_with_weight(
    *,
    title: str,
    section_path: list[str],
    important_keywords: list[str],
    content: str,
) -> str:
    weighted_parts = [title, *section_path, *important_keywords, content]
    return " ".join(part.strip() for part in weighted_parts if part and part.strip())


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[ :|\r\n\t,，。？?/`!！&^%()\[\]{}<>*~'\"\\]+", " ", text).strip()


def _similarity(query_weights: dict[str, float], document_weights: dict[str, float]) -> float:
    score = 1e-9
    query_total = 1e-9
    for token, weight in query_weights.items():
        if token in document_weights:
            score += weight
        query_total += weight
    return score / query_total


def _loads_jsonish(raw: str) -> dict:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _unique(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _looks_like_noise_ngram(value: str) -> bool:
    return any(stop_word in value for stop_word in STOP_WORDS) or any(
        char in STOP_CHARS for char in value
    )
