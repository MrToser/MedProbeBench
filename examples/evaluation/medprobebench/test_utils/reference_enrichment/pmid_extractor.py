"""PMID extractor - extract PMID and URL from reference text."""
import re
from typing import Optional, List

class PMIDExtractor:
    # PMID matching patterns: support multiple formats
    PATTERNS = [
        r'PMID:\s*(\d{7,8})',  # PMID: 12345678
        r'PMID\s*(\d{7,8})',   # PMID 12345678
        r'PMID:(\d{7,8})'      # PMID:12345678
    ]
    
    # URL matching pattern
    URL_PATTERN = r'https?://[^\s<>"\')\]]+[^\s<>"\')\].,;:!?]'

    def extract(self, text: str) -> Optional[str]:
        """Extract PMID and return a 7-8 digit numeric string."""
        for pattern in self.PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                pmid = match.group(1)
                if pmid.isdigit() and 7 <= len(pmid) <= 8:
                    return pmid
        return None

    def extract_urls(self, text: str) -> List[str]:
        """Extract all URLs."""
        if not text:
            return []
        urls = re.findall(self.URL_PATTERN, text)
        # Clean possible trailing punctuation from URLs
        cleaned_urls = []
        for url in urls:
            # Remove common trailing punctuation
            url = url.rstrip('.,;:!?')
            if url:
                cleaned_urls.append(url)
        return cleaned_urls

    def categorize_urls(self, urls: List[str]) -> dict:
        """
        Classify URLs into pubmed, doi, and other categories.
        Returns: {"pubmed": [...], "doi": [...], "other": [...]}
        """
        result = {"pubmed": [], "doi": [], "other": []}
        
        for url in urls:
            url_lower = url.lower()
            if "pubmed" in url_lower or "ncbi.nlm.nih.gov" in url_lower:
                result["pubmed"].append(url)
            elif "doi.org" in url_lower:
                result["doi"].append(url)
            else:
                result["other"].append(url)
        
        return result
