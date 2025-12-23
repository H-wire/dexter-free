
from typing import List
from duckduckgo_search import DDGS
from .base import BaseSearcher, SearchResult

class DuckDuckGoSearcher(BaseSearcher):
    """A searcher that uses the DuckDuckGo search API."""

    def __init__(self):
        self.searcher = "DuckDuckGo"

    async def get_search_results(self, query: str, max_results: int) -> List[SearchResult]:
        """Search DuckDuckGo and return the results."""
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        
        return [
            SearchResult(
                title=r["title"],
                url=r["href"],
                searcher=self.searcher,
            )
            for r in results
        ]
