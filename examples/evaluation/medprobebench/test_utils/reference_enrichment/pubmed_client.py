"""PubMed API client - query literature metadata and DOI."""
import requests
import time
from typing import Optional, Dict
from threading import Lock

class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""
    def __init__(self, calls_per_second: float):
        self.rate = calls_per_second
        self.tokens = calls_per_second
        self.last_update = time.time()
        self.lock = Lock()

    def acquire(self):
        """Acquire a token, waiting if none are available."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1:
                sleep_time = (1 - self.tokens) / self.rate
                time.sleep(sleep_time)
                self.tokens = 0
            else:
                self.tokens -= 1

class PubMedClient:
    """PubMed E-utilities API client."""
    def __init__(self, base_url: str, rate_limit: int, timeout: int, max_retries: int):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(rate_limit)

    def get_metadata(self, pmid: str) -> Optional[Dict]:
        """Get metadata (including DOI) for a single PMID."""
        url = f"{self.base_url}/esummary.fcgi"
        params = {"db": "pubmed", "id": pmid, "retmode": "json"}

        for attempt in range(self.max_retries):
            try:
                self.rate_limiter.acquire()
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()

                data = response.json()
                result = data.get("result", {}).get(pmid, {})

                # Extract DOI
                doi = None
                for id_obj in result.get("articleids", []):
                    if id_obj.get("idtype") == "doi":
                        doi = id_obj.get("value")
                        break

                return {"doi": doi, "title": result.get("title"), "journal": result.get("source")}

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep((2 ** attempt) * 1)  # Exponential backoff
                else:
                    print(f"  ⚠️  PMID {pmid} API failed: {e}")
                    return None

        return None
