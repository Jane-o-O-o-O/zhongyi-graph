from app.services.ragflow_compat.query import (
    build_content_with_weight,
    candidate_search_keywords,
    entity_keywords,
    parse_query_rewrite,
    token_similarity,
    tokenize_query,
)


def test_tokenize_query_keeps_tcm_terms_and_short_ngrams():
    tokens = tokenize_query("失眠可以从心脾两虚证候分析吗？")

    assert "失眠" in tokens
    assert "心脾两虚" in tokens
    assert "证候" in tokens
    assert "心脾" in tokens


def test_entity_keywords_filters_ngram_noise_from_retrieval_tokens():
    keywords = tokenize_query("失眠可以从哪些证候分析？")

    assert entity_keywords(keywords) == ["失眠"]


def test_candidate_search_keywords_prefers_entities_over_ngram_noise():
    keywords = tokenize_query("失眠可以从哪些证候分析？")

    assert candidate_search_keywords(keywords) == ["失眠"]


def test_candidate_search_keywords_falls_back_to_compact_non_stop_terms():
    keywords = tokenize_query("便秘怎么处理？")

    assert candidate_search_keywords(keywords) == ["便秘"]


def test_token_similarity_uses_unigram_and_adjacent_bigram_overlap():
    score = token_similarity(
        "失眠 心脾两虚 归脾汤",
        [
            "失眠 心脾两虚 归脾汤",
            "柴胡 桂枝 干姜",
        ],
    )

    assert score[0] > 0.99
    assert score[1] < 0.01


def test_parse_query_rewrite_repairs_json_like_llm_output():
    parsed = parse_query_rewrite(
        """
        下面是结果：
        ```json
        {"answer_type_keywords":["Syndrome"],"entities_from_query":["失眠","心脾两虚","归脾汤","党参","脾","心"]}
        ```
        """
    )

    assert parsed.answer_type_keywords == ["Syndrome"]
    assert parsed.entities_from_query == ["失眠", "心脾两虚", "归脾汤", "党参", "脾"]


def test_build_content_with_weight_prioritizes_title_keywords_and_body():
    content = build_content_with_weight(
        title="失眠",
        section_path=["内科", "不寐"],
        important_keywords=["心脾两虚", "归脾汤"],
        content="失眠可辨为心脾两虚，常用归脾汤。",
    )

    assert content.startswith("失眠 内科 不寐 心脾两虚 归脾汤 ")
    assert content.endswith("失眠可辨为心脾两虚，常用归脾汤。")
