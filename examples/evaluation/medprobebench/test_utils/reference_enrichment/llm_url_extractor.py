"""
LLM-based URL Extractor - extract URL information from citation text using an LLM.

Features:
1. Extract PMID and DOI from citation text
2. Construct valid PubMed and DOI URLs
3. Handle various non-standard citation formats
"""

import json
import re
import os
from typing import Dict, Optional
from pathlib import Path
from openai import OpenAI


EXTRACTION_PROMPT = """You are a medical literature reference parser. Extract structured information from the given reference text.

## Task
Parse the reference and extract:
1. **PMID** - PubMed ID (7-8 digit number)
2. **DOI** - Digital Object Identifier (format: 10.xxxx/xxxxx)
3. **Any existing URLs** in the text (including website links, online resources, etc.)

## Input Reference
{reference_text}

## Output Format
Return a JSON object with the following structure:
{{
    "pmid": "12345678" or null,
    "doi": "10.1000/xyz123" or null,
    "existing_urls": ["url1", "url2"] or [],
    "other_urls": ["non-pubmed-non-doi-urls"] or [],
    "confidence": "high" | "medium" | "low"
}}

## Rules
1. PMID must be 7-8 digits only
2. DOI must start with "10." 
3. existing_urls should contain ALL URLs found in the text
4. other_urls should contain URLs that are NOT PubMed or DOI links (e.g., organization websites, guideline pages, etc.)
5. If no identifiers found, return null for those fields
6. Be conservative - only extract if confident

Return ONLY the JSON object, no explanation."""


class LLMURLExtractor:
    """Use an LLM to extract URL information from citation text."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model
        
        if not self.api_key:
            raise ValueError("API key required. Set via parameter or OPENAI_API_KEY env var.")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
    
    def extract(self, reference_text: str) -> Dict:
        """
        Extract URL information from citation text.
        
        Returns:
            {
                "pmid": str or None,
                "doi": str or None,
                "pubmed_url": str or None,
                "doi_url": str or None,
                "existing_urls": list,
                "other_urls": list,  # Added: other URLs that are not PubMed/DOI
                "confidence": str
            }
        """
        result = {
            "pmid": None,
            "doi": None,
            "pubmed_url": None,
            "doi_url": None,
            "existing_urls": [],
            "other_urls": [],  # Added
            "confidence": "none",
        }
        
        if not reference_text or not reference_text.strip():
            return result
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise medical literature parser. Return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": EXTRACTION_PROMPT.format(reference_text=reference_text)
                    }
                ],
                temperature=0,
                max_tokens=500,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean JSON (remove possible markdown code fence markers)
            content = self._clean_json_response(content)
            
            parsed = json.loads(content)
            
            # Validate and populate result
            pmid = parsed.get("pmid")
            if pmid and self._is_valid_pmid(str(pmid)):
                result["pmid"] = str(pmid)
                result["pubmed_url"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
            
            doi = parsed.get("doi")
            if doi and self._is_valid_doi(str(doi)):
                result["doi"] = str(doi)
                result["doi_url"] = f"https://doi.org/{doi}"
            
            result["existing_urls"] = parsed.get("existing_urls", [])
            result["other_urls"] = parsed.get("other_urls", [])
            result["confidence"] = parsed.get("confidence", "low")
            
        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON parse error: {e}")
        except Exception as e:
            print(f"  ⚠️  LLM extraction error: {e}")
        
        return result
    
    def _clean_json_response(self, content: str) -> str:
        """Clean JSON content returned by LLM."""
        # Remove markdown code fence markers
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'^```\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        return content.strip()
    
    def _is_valid_pmid(self, pmid: str) -> bool:
        """Validate PMID format."""
        return bool(pmid and pmid.isdigit() and 7 <= len(pmid) <= 8)
    
    def _is_valid_doi(self, doi: str) -> bool:
        """Validate DOI format."""
        return bool(doi and doi.startswith("10.") and "/" in doi)


# Test code
if __name__ == "__main__":
    # Test samples
    test_refs = [
        "Smith J, et al. A study on diabetes. J Med. 2020;15:100-110. PMID: 12345678. doi: 10.1000/example",
        "Brown A. Cancer research findings. Nature. 2019. https://pubmed.ncbi.nlm.nih.gov/87654321/",
        "No identifiers in this reference. Some Journal. 2021;1:1-10.",
        "NCCN Guidelines. Available at: https://www.nccn.org/guidelines/category_1. Accessed 2024.",
    ]
    
    try:
        extractor = LLMURLExtractor()
        for ref in test_refs:
            print(f"\nInput: {ref[:80]}...")
            result = extractor.extract(ref)
            print(f"Result: {json.dumps(result, indent=2)}")
    except ValueError as e:
        print(f"Error: {e}")
