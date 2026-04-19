"""Cache manager - avoid duplicate API calls."""
import json
from pathlib import Path
from typing import Optional, Dict
from threading import Lock

class CacheManager:
    """Simple JSON file cache."""
    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.cache = self._load()
        self.lock = Lock()

    def _load(self) -> Dict:
        """Load cache."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def get(self, pmid: str) -> Optional[Dict]:
        """Get from cache."""
        with self.lock:
            return self.cache.get(pmid)

    def set(self, pmid: str, data: Dict):
        """Save to cache."""
        with self.lock:
            self.cache[pmid] = data

    def save(self):
        """Persist cache."""
        with self.lock:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
