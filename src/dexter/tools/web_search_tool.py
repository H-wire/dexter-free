
from typing import List
from dexter.tools.web_search.duckduckgo import DuckDuckGoSearcher

async def web_search(query: str, max_results: int = 5) -> List[dict]:
    """
    Performs a web search using DuckDuckGo and returns the results.

    Args:
        query: The search query.
        max_results: The maximum number of results to return.

    Returns:
        A list of search results, each with a title and URL.
    """
    searcher = DuckDuckGoSearcher()
    results = await searcher.get_search_results(query, max_results)
    return [result.dict() for result in results]
