from services.search.ranking import rank_search_results

SAMPLE_RESULTS = [
    {
        "title": "Canada - AP News",
        "url": "https://apnews.com/hub/canada",
        "snippet": "Latest Canada news coverage from AP News.",
    },
    {
        "title": "CBC News - Canada",
        "url": "https://www.cbc.ca/news/canada",
        "snippet": "Your source for Canadian news in English.",
    },
]


def test_rank_search_results_handles_none_query():
    # query feeds re.findall(..., query) and query.lower(); a None query made
    # both raise. A non-string query should behave as "no query terms".
    assert rank_search_results(None, []) == []


def test_rank_search_results_handles_non_string_query():
    out = rank_search_results(123, SAMPLE_RESULTS)
    assert isinstance(out, list)
    assert len(out) == len(SAMPLE_RESULTS)


def test_rank_search_results_string_query_still_ranks():
    out = rank_search_results("Canada news today", SAMPLE_RESULTS)
    assert isinstance(out, list)
    assert len(out) == len(SAMPLE_RESULTS)
    assert {item["url"] for item in out} == {r["url"] for r in SAMPLE_RESULTS}
