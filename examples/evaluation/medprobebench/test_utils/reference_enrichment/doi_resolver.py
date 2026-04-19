"""DOI resolver - get publisher URL from DOI."""
import requests
from typing import Optional

class DOIResolver:
    """DOI resolver - follow redirects to get publisher URL."""
    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def resolve(self, doi: str) -> Optional[str]:
        """Resolve DOI to the publisher URL."""
        if not doi:
            return None

        try:
            url = f"https://doi.org/{doi}"
            # Follow redirects to get final URL
            response = requests.get(url, timeout=self.timeout, allow_redirects=True)
            if response.status_code == 200:
                return response.url
        except Exception as e:
            print(f"  ⚠️  DOI {doi} resolve failed: {e}")

        return None
