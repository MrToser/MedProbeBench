#!/usr/bin/env python3
"""
Reference Enrichment - Add accessible URLs to medical literature references.
Supports two modes:
1. API mode: Uses PubMed API (requires PMID)
2. LLM mode: Uses LLM for intelligent extraction (more comprehensive)

Usage:
    python enrich_references.py -i ./extracted_claims/ -o ./enriched_claims/
    python enrich_references.py -i ./extracted_claims/ -o ./enriched_claims/ --use-llm
"""
import json
import sys
import os  # Add this import
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add path
sys.path.insert(0, str(Path(__file__).parent))

from reference_enrichment.pmid_extractor import PMIDExtractor
from reference_enrichment.pubmed_client import PubMedClient
from reference_enrichment.doi_resolver import DOIResolver
from reference_enrichment.cache_manager import CacheManager

# Import md_utils (for processing MD files)
from md_utils import extract_references_section, parse_references


@dataclass
class EnrichmentStats:
    """Enrichment statistics"""
    total: int = 0
    complete: int = 0
    partial: int = 0
    no_pmid: int = 0
    failed: int = 0


class ReferenceEnricher:
    """Reference enrichment processor"""
    
    def __init__(
        self,
        cache_dir: Path,
        workers: int = 10,
        resolve_publisher: bool = True,
        use_llm: bool = False,
        llm_config: dict = None,
    ):
        self.workers = workers
        self.resolve_publisher = resolve_publisher
        self.use_llm = use_llm
        
        # Initialize components
        self.pmid_extractor = PMIDExtractor()
        self.cache = CacheManager(cache_dir / 'reference_cache.json')
        
        # Load configuration
        config_path = Path(__file__).parent / "reference_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        pubmed_config = self.config.get("pubmed", {})
        self.pubmed_client = PubMedClient(
            base_url=pubmed_config.get("base_url", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"),
            rate_limit=pubmed_config.get("rate_limit", 3),
            timeout=pubmed_config.get("timeout", 30),
            max_retries=pubmed_config.get("max_retries", 3),
        )
        
        self.doi_resolver = DOIResolver() if resolve_publisher else None
        
        # LLM extractor (initialized on demand)
        self.llm_extractor = None
        if use_llm:
            self._init_llm_extractor(llm_config)
    
    def _init_llm_extractor(self, llm_config: dict = None):
        """Initialize LLM extractor"""
        from reference_enrichment.llm_url_extractor import LLMURLExtractor
        
        gpt_config = self.config.get("gpt", {})
        config = llm_config or {}
        
        self.llm_extractor = LLMURLExtractor(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=config.get("model") or os.getenv("OPENAI_MODEL", "gpt-4.1"),
        )

    def _extract_other_urls(self, ref_text: str, exclude_urls: list = None) -> list:
        """
        Extract non-PubMed and non-DOI URLs from reference text.
        
        Args:
            ref_text: Reference text
            exclude_urls: List of URLs to exclude (e.g., already identified pubmed_url, doi_url, publisher_url)
        
        Returns:
            List of other URLs
        """
        exclude_urls = exclude_urls or []
        exclude_set = set(url.lower() for url in exclude_urls if url)
        
        # Extract all URLs
        all_urls = self.pmid_extractor.extract_urls(ref_text)
        
        # Categorize URLs
        categorized = self.pmid_extractor.categorize_urls(all_urls)
        
        # Collect other URLs (excluding already identified ones)
        other_urls = []
        for url in categorized.get("other", []):
            if url.lower() not in exclude_set:
                other_urls.append(url)
        
        # Also include URLs from pubmed and doi categories that were not identified (possibly non-standard formats)
        for url in categorized.get("pubmed", []) + categorized.get("doi", []):
            if url.lower() not in exclude_set:
                # Check if a similar URL already exists in the exclude set
                is_duplicate = False
                for excluded in exclude_set:
                    if url.lower() in excluded or excluded in url.lower():
                        is_duplicate = True
                        break
                if not is_duplicate:
                    other_urls.append(url)
        
        return list(set(other_urls))  # Deduplicate
        

    def enrich_reference(self, ref_num: str, ref_text: str) -> dict:
        """Process a single reference"""
        result = {
            "text": ref_text,
            "pmid": None,
            "doi": None,
            "urls": {
                "pubmed": None,
                "doi": None,
                "publisher": None,
                "other": None  # New field: stores other URLs
            },
            "metadata": {"status": "pending", "last_updated": datetime.now().isoformat()}
        }

        # Method 1: Use LLM for extraction
        if self.use_llm and self.llm_extractor:
            llm_result = self.llm_extractor.extract(ref_text)
            
            result["pmid"] = llm_result.get("pmid")
            result["doi"] = llm_result.get("doi")
            result["urls"]["pubmed"] = llm_result.get("pubmed_url")
            result["urls"]["doi"] = llm_result.get("doi_url")
            result["metadata"]["extraction_method"] = "llm"
            result["metadata"]["confidence"] = llm_result.get("confidence", "none")
            
            # If LLM found a DOI, try to resolve the publisher URL
            if result["doi"] and self.resolve_publisher and self.doi_resolver:
                publisher_url = self.doi_resolver.resolve(result["doi"])
                if publisher_url:
                    result["urls"]["publisher"] = publisher_url
            
            # Extract other URLs (excluding already identified ones)
            exclude_urls = [
                result["urls"]["pubmed"],
                result["urls"]["doi"],
                result["urls"]["publisher"],
            ]
            # Also add existing_urls returned by LLM (to avoid duplicates)
            existing_urls = llm_result.get("existing_urls", [])
            
            other_urls = self._extract_other_urls(ref_text, exclude_urls)
            # Merge LLM-found existing_urls and regex-extracted other_urls
            all_other_urls = list(set(existing_urls + other_urls))
            # Filter out already identified URLs again
            final_other_urls = []
            for url in all_other_urls:
                url_lower = url.lower()
                is_excluded = False
                for ex_url in exclude_urls:
                    if ex_url and (url_lower == ex_url.lower() or url_lower in ex_url.lower() or ex_url.lower() in url_lower):
                        is_excluded = True
                        break
                if not is_excluded:
                    final_other_urls.append(url)
            
            if len(final_other_urls) == 0:
                final_other_urls = ""
            else:
                final_other_urls = final_other_urls[0]
            
            result["urls"]["other"] = final_other_urls
            
            if result["pmid"] or result["doi"]:
                result["metadata"]["status"] = "complete"
            elif final_other_urls:
                result["metadata"]["status"] = "partial_with_urls"
            else:
                result["metadata"]["status"] = "no_identifiers"
            
            return result

        # Method 2: Traditional approach - regex extract PMID + PubMed API
        pmid = self.pmid_extractor.extract(ref_text)
        
        # Extract all URLs (regardless of whether PMID was found)
        all_urls = self.pmid_extractor.extract_urls(ref_text)
        categorized_urls = self.pmid_extractor.categorize_urls(all_urls)
        
        if not pmid:
            result["metadata"]["status"] = "no_pmid"
            result["metadata"]["extraction_method"] = "regex"
            # Even without PMID, save any other URLs found
            result["urls"]["other"] = categorized_urls.get("other", [])
            # If there are other URLs, update the status
            if result["urls"]["other"]:
                result["metadata"]["status"] = "no_pmid_with_urls"
            return result

        result["pmid"] = pmid
        result["urls"]["pubmed"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"

        # Check cache
        cached = self.cache.get(pmid)
        if cached:
            result["doi"] = cached.get("doi")
            result["urls"]["doi"] = f"https://doi.org/{cached['doi']}" if cached.get("doi") else None
            result["urls"]["publisher"] = cached.get("publisher_url")
            result["urls"]["other"] = cached.get("other_urls", [])
            result["metadata"]["status"] = cached.get("status", "complete")
            result["metadata"]["data_source"] = "cache"
            return result

        # Call PubMed API
        metadata = self.pubmed_client.get_metadata(pmid)
        if metadata and metadata.get("doi"):
            result["doi"] = metadata["doi"]
            result["urls"]["doi"] = f"https://doi.org/{metadata['doi']}"

            # Resolve publisher URL
            if self.resolve_publisher and self.doi_resolver:
                publisher_url = self.doi_resolver.resolve(metadata["doi"])
                if publisher_url:
                    result["urls"]["publisher"] = publisher_url
                    result["metadata"]["status"] = "complete"
                else:
                    result["metadata"]["status"] = "partial"
            else:
                result["metadata"]["status"] = "partial"
        else:
            result["metadata"]["status"] = "partial"

        # Extract other URLs (excluding already identified ones)
        exclude_urls = [
            result["urls"]["pubmed"],
            result["urls"]["doi"],
            result["urls"]["publisher"],
        ]
        result["urls"]["other"] = self._extract_other_urls(ref_text, exclude_urls)

        # Save to cache
        self.cache.set(pmid, {
            "doi": result["doi"],
            "publisher_url": result["urls"]["publisher"],
            "other_urls": result["urls"]["other"],
            "status": result["metadata"]["status"]
        })

        result["metadata"]["data_source"] = "pubmed_api"
        return result

    def enrich_file(self, input_file: Path, output_file: Path) -> EnrichmentStats:
        """
        Process a single file (supports JSON and MD).
        
        - JSON: Reads the reference field directly
        - MD: Uses md_utils to extract references
        """
        print(f"\n📄 Processing: {input_file.name}")
        
        # Choose processing method based on file type
        if input_file.suffix.lower() == '.md':
            return self._enrich_markdown_file(input_file, output_file)
        else:
            return self._enrich_json_file(input_file, output_file)
    
    def _enrich_markdown_file(self, input_file: Path, output_file: Path) -> EnrichmentStats:
        """Process a Markdown file"""
        md_text = input_file.read_text(encoding='utf-8')
        
        # Use md_utils to extract references
        _, ref_section = extract_references_section(md_text)
        
        if not ref_section:
            print("   ⚠️ No references found in MD file, skipping enrichment")
            # Copy the original file directly
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(md_text, encoding='utf-8')
            return EnrichmentStats()
        
        # Parse references
        references = parse_references(ref_section)
        
        if not references:
            print("   ⚠️ Could not parse references, skipping enrichment")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(md_text, encoding='utf-8')
            return EnrichmentStats()
        
        print(f"   Found {len(references)} references")
        
        # Process each reference (reuse JSON logic)
        need_processing = [(num, text) for num, text in references.items() if text.strip()]
        
        if not need_processing:
            print("   ✅ All references are empty")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(md_text, encoding='utf-8')
            return EnrichmentStats()
        
        print(f"   Processing {len(need_processing)} non-empty references...")
        
        # Enrich references
        stats = EnrichmentStats(total=len(need_processing))
        enriched = {}
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self.enrich_reference, num, text): num 
                for num, text in need_processing
            }
            
            with tqdm(total=len(need_processing), desc=f"   {input_file.stem}", leave=False) as pbar:
                for future in as_completed(futures):
                    ref_num = futures[future]
                    try:
                        result = future.result()
                        enriched[ref_num] = result
                        
                        status = result.get("metadata", {}).get("status", "failed")
                        if status == "complete":
                            stats.complete += 1
                        elif status in ["partial", "partial_with_urls", "no_pmid_with_urls"]:
                            stats.partial += 1
                        elif status in ["no_pmid", "no_identifiers"]:
                            stats.no_pmid += 1
                        else:
                            stats.failed += 1
                    except Exception as e:
                        print(f"\n      ❌ Ref {ref_num} failed: {e}")
                        stats.failed += 1
                    finally:
                        pbar.update(1)
        
        # Save in JSON format (MD is converted to JSON after enrichment)
        output_data = {
            "source": input_file.name,
            "references": enriched
        }
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        # Force output as .json
        if output_file.suffix.lower() == '.md':
            output_file = output_file.with_suffix('.json')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ Saved: {output_file.name} (converted to JSON)")
        print(f"      Complete: {stats.complete}, Partial: {stats.partial}, No ID: {stats.no_pmid}")
        
        return stats
    
    def _enrich_json_file(self, input_file: Path, output_file: Path) -> EnrichmentStats:
        """Process a JSON file (original logic)"""
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"   Found {len(data.get('reference', data.get('references', {})))} references")

        # Support both key names
        references = data.get("reference", data.get("references", {}))
        ref_key_used = "reference" if "reference" in data else "references"
        
        if not references:
            print("   ⚠️ No references found, skipping enrichment")
            # Save original data without any modification
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return EnrichmentStats()

        # Identify references that need processing - add text validation
        need_processing = []
        for ref_num, ref_data in references.items():
            if isinstance(ref_data, str):
                # String type: only process if non-empty
                if ref_data.strip():
                    need_processing.append((ref_num, ref_data))
                else:
                    print(f"   ⚠️ Ref {ref_num} is empty string, skipping")
            elif isinstance(ref_data, dict):
                # Dict type: check if text exists
                ref_text = ref_data.get("text", "").strip()
                if not ref_text:
                    print(f"   ⚠️ Ref {ref_num} has no text, skipping")
                    continue
                
                # Check if reprocessing is needed
                status = ref_data.get("metadata", {}).get("status", "pending")
                has_urls = ref_data.get("urls", {}).get("pubmed") or ref_data.get("urls", {}).get("doi")
                if status == "pending" or (status in ["no_pmid", "no_identifiers"] and self.use_llm and not has_urls):
                    need_processing.append((ref_num, ref_text))

        if not need_processing:
            print("   ✅ All references already processed or empty")
            # Still need to save the file (some references may have been processed)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return EnrichmentStats()

        print(f"   Processing {len(need_processing)} non-empty references...")

        # Process references
        stats = EnrichmentStats(total=len(need_processing))
        enriched = {}
        
        # Use concurrent processing
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self.enrich_reference, num, text): num 
                for num, text in need_processing
            }

            with tqdm(total=len(need_processing), desc=f"   {input_file.stem}", leave=False) as pbar:
                for future in as_completed(futures):
                    ref_num = futures[future]
                    try:
                        result = future.result()
                        enriched[ref_num] = result
                        
                        status = result.get("metadata", {}).get("status", "failed")
                        if status == "complete":
                            stats.complete += 1
                        elif status in ["partial", "partial_with_urls", "no_pmid_with_urls"]:
                            stats.partial += 1
                        elif status in ["no_pmid", "no_identifiers"]:
                            stats.no_pmid += 1
                        else:
                            stats.failed += 1
                    except Exception as e:
                        print(f"\n      ❌ Ref {ref_num} failed: {e}")
                        stats.failed += 1
                    finally:
                        pbar.update(1)

        # Update data
        for ref_num, ref_data in enriched.items():
            references[ref_num] = ref_data
        
        data[ref_key_used] = references

        # Save results
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Count other URLs
        other_urls_count = sum(
            len(ref_data.get("urls", {}).get("other", []))
            for ref_data in enriched.values()
            if isinstance(ref_data, dict)
        )

        print(f"   ✅ Saved: {output_file.name}")
        print(f"      Complete: {stats.complete}, Partial: {stats.partial}, No ID: {stats.no_pmid}")
        print(f"      Other URLs found: {other_urls_count}")

        return stats
