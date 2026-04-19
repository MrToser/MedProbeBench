# -*- coding: utf-8 -*-
"""Reference recall evaluation module.

Evaluates whether GT references are recalled by model-predicted references.
Supports sync and async evaluation.

Logic:
1. Collect GT and predicted URLs by section
2. Fetch URL contents from shared cache
3. Use LLM to judge whether two URL contents refer to the same underlying source/topic
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse, parse_qs

from .cache_urls_jina import get_url_cache, URLCacheManager

if TYPE_CHECKING:
    from openai import OpenAI, AsyncOpenAI
    from ..task_success_rate import GTClaim, PredClaim


# ============================================================================
# LLM Prompt Template (Optimized for fewer tokens)
# ============================================================================

REFERENCE_CONTENT_MATCH_PROMPT = """Determine whether the following two URLs refer to the same underlying topic, work, or object.

GT URL: {gt_url}
GT Content (excerpt):
{gt_content}

Pred URL: {pred_url}
Pred Content (excerpt):
{pred_content}

Consider it a match if both contents discuss the same underlying thing (e.g., the same study, dataset, method, system, or phenomenon), even if their wording, focus, conclusions, or level of detail differ.

If one or both contents are mainly meta-information (e.g., titles, abstracts, or repository pages), use available clues (such as names or referenced objects) and apply a slightly more permissive judgment.

Ignore minor differences or disagreements.

Return JSON only:
{{"is_match": true/false, "reason": "brief explanation of the shared underlying object"}}
"""

# ============================================================================
# Utility Functions
# ============================================================================

def parse_json_to_dict(json_string: str) -> dict:
    """Extract first valid JSON object from string."""
    if not json_string:
        return {}
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", json_string.strip(), flags=re.IGNORECASE
    )
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


def extract_url_from_reference(ref_value: Any) -> str:
    """Extract best URL from reference value.
    
    Supports:
    1. Old format: URL string
    2. New format: dict with urls field
    """
    if isinstance(ref_value, str):
        return ref_value
    
    if isinstance(ref_value, dict):
        urls = ref_value.get("urls", {})
        if isinstance(urls, dict):
            for key in ["doi", "pubmed", "publisher", "other"]:
                url = urls.get(key)
                if url:
                    return url
        
        doi = ref_value.get("doi")
        if doi:
            return f"https://doi.org/{doi}"
        
        pmid = ref_value.get("pmid")
        if pmid:
            return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"

        other_urls = ref_value.get("other", [])  
        if isinstance(other_urls, list) and other_urls:
            return other_urls[0]
    
    return ""


# ============================================================================
# Reference Recall Evaluator
# ============================================================================

class ReferenceRecallEvaluator:
    """Reference recall evaluator (sync and async support)."""

    def __init__(
        self,
        client: "OpenAI | AsyncOpenAI",
        grader_model: str,
        verbose: bool = True,
        fetch_timeout: int = 60,
        max_content_length: int = 3000,
        url_cache: URLCacheManager | None = None,
        max_concurrent: int = 10,
    ) -> None:
        self.client = client
        self.grader_model = grader_model
        self.verbose = verbose
        self.max_content_length = max_content_length
        self.max_concurrent = max_concurrent
        self._semaphore: asyncio.Semaphore | None = None
        
        self._url_cache = url_cache or get_url_cache(
            fetch_timeout=fetch_timeout,
            max_content_length=max_content_length,
            verbose=verbose,
        )
        
        # Stats
        self.grader_calls = 0
        self.grader_total_time = 0.0
        self.content_match_count = 0
        
        # Token usage
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Lazy init semaphore (must be created in event loop)."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    # ========================================================================
    # LLM Call Methods
    # ========================================================================

    async def _call_grader_async(self, prompt: str) -> str:
        """Async call to grader model."""
        messages = [
            {"role": "system", "content": "You are a reference matcher. Return JSON only."},
            {"role": "user", "content": prompt},
        ]

        semaphore = self._get_semaphore()
        async with semaphore:
            start_time = time.perf_counter()
            result = ""
            
            for attempt in range(3):
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.grader_model,
                        messages=messages,
                        temperature=0,
                        max_tokens=100,  # Reduced from 500
                    )
                    result = resp.choices[0].message.content or ""
                    if hasattr(resp, 'usage') and resp.usage:
                        self.prompt_tokens += resp.usage.prompt_tokens or 0
                        self.completion_tokens += resp.usage.completion_tokens or 0
                    break
                except TypeError:
                    loop = asyncio.get_event_loop()
                    resp = await loop.run_in_executor(
                        None,
                        lambda: self.client.chat.completions.create(
                            model=self.grader_model,
                            messages=messages,
                            temperature=0,
                            max_tokens=100,
                        )
                    )
                    result = resp.choices[0].message.content or ""
                    if hasattr(resp, 'usage') and resp.usage:
                        self.prompt_tokens += resp.usage.prompt_tokens or 0
                        self.completion_tokens += resp.usage.completion_tokens or 0
                    break
                except Exception as e:
                    if attempt == 2:
                        if self.verbose:
                            print(f"      ⚠️ Grader call failed: {e}")
                        result = ""
                    else:
                        await asyncio.sleep(1 * (attempt + 1))
                    continue
            
            elapsed = time.perf_counter() - start_time
            self.grader_calls += 1
            self.grader_total_time += elapsed
            
            return result

    # ========================================================================
    # Reference Matching Core Logic
    # ========================================================================

    def _extract_middle_content(self, content: str, max_length: int) -> str:
        """Extract the core middle portion from cached content (skip the beginning, focus on the body)."""
        if not content or len(content) <= max_length:
            return content
        
        skip_ratio = 0
        start_pos = int(len(content) * skip_ratio)
        end_pos = start_pos + max_length
        
        return content[start_pos:end_pos]

    async def _match_references_async(
        self, 
        gt_url: str, 
        pred_url: str,
        url_contents: dict[str, str],
    ) -> dict:
        """Async check if two references match (content-based only)."""
        
        gt_content = url_contents.get(gt_url, "")
        pred_content = url_contents.get(pred_url, "")
        
        if not gt_content and not pred_content:
            return {
                "is_match": False,
                "match_method": "content_unavailable",
                "reason": "Both URLs unavailable",
                "gt_url": gt_url,
                "pred_url": pred_url,
            }
        
        if not gt_content:
            return {
                "is_match": False,
                "match_method": "gt_content_unavailable",
                "reason": "GT URL unavailable",
                "gt_url": gt_url,
                "pred_url": pred_url,
            }
        
        if not pred_content:
            return {
                "is_match": False,
                "match_method": "pred_content_unavailable", 
                "reason": "Pred URL unavailable",
                "gt_url": gt_url,
                "pred_url": pred_url,
            }
        
        # Extract the core middle portion from cached content
        gt_content_middle = self._extract_middle_content(gt_content, self.max_content_length)
        pred_content_middle = self._extract_middle_content(pred_content, self.max_content_length)
        
        prompt = REFERENCE_CONTENT_MATCH_PROMPT.format(
            gt_url=gt_url,
            gt_content=gt_content_middle,
            pred_url=pred_url,
            pred_content=pred_content_middle,
        )
        
        result_text = await self._call_grader_async(prompt)
        parsed = parse_json_to_dict(result_text)
        
        is_match = parsed.get("is_match", False)
        reason = parsed.get("reason", "LLM judgment")
        
        if is_match:
            self.content_match_count += 1
        
        return {
            "is_match": is_match,
            "match_method": "content_llm",
            "reason": reason,
            "gt_url": gt_url,
            "pred_url": pred_url,
        }

    # ========================================================================
    # Section-level Evaluation
    # ========================================================================

    def _collect_references_by_section(
        self,
        claims: list,
        references: dict[str, Any],
    ) -> dict[str, dict[str, str]]:
        """Collect references by section."""
        section_refs = {}
        for claim in claims:
            section = claim.section or "Unknown"
            if section not in section_refs:
                section_refs[section] = {}
            
            for ref_id in claim.reference:
                if ref_id.startswith('http'):
                    section_refs[section][ref_id] = ref_id
                elif ref_id in references:
                    url = extract_url_from_reference(references[ref_id])
                    if url:
                        section_refs[section][ref_id] = url
        
        return section_refs

    async def _evaluate_section_async(
        self,
        gt_section_refs: dict[str, str],
        pred_section_refs: dict[str, str],
        section: str,
        url_contents: dict[str, str],
    ) -> dict:
        """Async evaluate single section's reference recall."""
        if not gt_section_refs:
            return {
                "section": section,
                "gt_references": [],
                "pred_references": list(pred_section_refs.keys()),
                "recall": 1.0,
                "precision": 1.0 if not pred_section_refs else 0.0,
                "matched_refs": [],
                "total_gt_refs": 0,
                "total_pred_refs": len(pred_section_refs),
                "matched_count": 0,
                "match_details": [],
                "explanation": f"Section '{section}' has no GT references",
            }

        if not pred_section_refs:
            return {
                "section": section,
                "gt_references": list(gt_section_refs.keys()),
                "pred_references": [],
                "recall": 0.0,
                "precision": 0.0,
                "matched_refs": [],
                "total_gt_refs": len(gt_section_refs),
                "total_pred_refs": 0,
                "matched_count": 0,
                "match_details": [],
                "explanation": f"Section '{section}' has no predicted references",
            }

        gt_ref_list = list(gt_section_refs.items())
        pred_ref_list = list(pred_section_refs.items())

        matched_gt_refs = []
        matched_pred_indices = set()
        match_details = []
        
        for gt_ref_id, gt_url in gt_ref_list:
            for i, (pred_ref_id, pred_url) in enumerate(pred_ref_list):
                if i in matched_pred_indices:
                    continue
                match_result = await self._match_references_async(gt_url, pred_url, url_contents)
                
                if match_result["is_match"]:
                    matched_gt_refs.append(gt_ref_id)
                    matched_pred_indices.add(i)
                    match_details.append({
                        "gt_ref_id": gt_ref_id,
                        "gt_url": gt_url,
                        "pred_ref_id": pred_ref_id,
                        "pred_url": pred_url,
                        "method": match_result["match_method"],
                        "reason": match_result["reason"],
                    })
                    if self.verbose:
                        print(f"        ✓ Match: [{gt_ref_id}] <-> [{pred_ref_id}] ({match_result['match_method']})")
                    break
                
                
        recall = len(matched_gt_refs) / len(gt_ref_list) if gt_ref_list else 0.0
        precision = len(matched_gt_refs) / len(pred_ref_list) if pred_ref_list else 0.0

        return {
            "section": section,
            "gt_references": list(gt_section_refs.keys()),
            "pred_references": list(pred_section_refs.keys()),
            "recall": recall,
            "precision": precision,
            "matched_refs": matched_gt_refs,
            "total_gt_refs": len(gt_ref_list),
            "total_pred_refs": len(pred_ref_list),
            "matched_count": len(matched_gt_refs),
            "match_details": match_details,
            "explanation": f"Section '{section}' matched {len(matched_gt_refs)}/{len(gt_ref_list)} GT refs",
        }

    # ========================================================================
    # Main Evaluation Entry
    # ========================================================================

    async def evaluate(
        self,
        gt_claims: list,
        pred_claims: list,
        gt_references: dict[str, str],
        pred_references: dict[str, str],
    ) -> dict:
        """Async evaluate reference recall with quantity bonus (main entry)."""
        start_time = time.perf_counter()
        
        # Collect references by section
        gt_refs_by_section = self._collect_references_by_section(gt_claims, gt_references)
        pred_refs_by_section = self._collect_references_by_section(pred_claims, pred_references)
        
        # Collect all URLs and prefetch contents
        gt_urls = set()
        pred_urls = set()
        for refs in gt_refs_by_section.values():
            gt_urls.update(refs.values())
        for refs in pred_refs_by_section.values():
            pred_urls.update(refs.values())

        all_urls = gt_urls | pred_urls

        if self.verbose:
            print(f"    Prefetching URLs (using shared cache):")
            print(f"      GT URLs: {len(gt_urls)}")
            print(f"      Pred URLs: {len(pred_urls)}")
            print(f"      Total unique URLs: {len(all_urls)}")
            print(f"      Overlapping URLs: {len(gt_urls & pred_urls)}")
        
        # Async batch fetch
        url_contents = await self._url_cache.get_batch_async(list(all_urls))
        
        section_results = []
        section_stats = {}
        total_matched = 0
        total_gt = 0
        total_pred = 0
        
        all_sections = set(gt_refs_by_section.keys()) | set(pred_refs_by_section.keys())
        
        if self.verbose:
            print(f"    Evaluating {len(all_sections)} sections for reference recall...")
        
        # Parallel evaluate all sections
        section_tasks = []
        section_list = sorted(all_sections)
        
        for section in section_list:
            gt_section_refs = gt_refs_by_section.get(section, {})
            pred_section_refs = pred_refs_by_section.get(section, {})
            
            if self.verbose:
                print(f"      Section '{section}': GT={len(gt_section_refs)}, Pred={len(pred_section_refs)}")
            
            section_tasks.append(
                self._evaluate_section_async(gt_section_refs, pred_section_refs, section, url_contents)
            )
        
        section_results = await asyncio.gather(*section_tasks)
        
        for result in section_results:
            total_matched += result["matched_count"]
            total_gt += result["total_gt_refs"]
            total_pred += result["total_pred_refs"]
            
            section_stats[result["section"]] = {
                "total_gt_refs": result["total_gt_refs"],
                "total_pred_refs": result["total_pred_refs"],
                "matched_refs": result["matched_count"],
                "recall": result["recall"],
                "precision": result["precision"],
            }
        
        # Calculate base recall
        base_recall = total_matched / total_gt if total_gt > 0 else 0.0
        overall_precision = total_matched / total_pred if total_pred > 0 else 0.0
        
        quantity_score = min(1.0, total_pred / total_gt) if total_gt > 0 else 0.0
        overall_recall = quantity_score * 0.4 + base_recall * 0.6 if total_gt > 0 else 0.0
        
        elapsed_time = time.perf_counter() - start_time
        
        cache_stats = self._url_cache.get_stats()
        
        if self.verbose:
            print(f"  ⏱️  Reference recall evaluation done: {elapsed_time:.2f}s")
            print(f"      GT refs: {total_gt}, Pred refs: {total_pred}, Matched: {total_matched}")
            print(f"      Base recall: {base_recall:.2%}")
            print(f"      Quantity score: {quantity_score:.2%}")
            print(f"      Final recall score: {overall_recall:.2%} (40% quantity + 60% recall)")
            print(f"      Content match: {self.content_match_count}")
            print(f"      Cache hits: memory={cache_stats['memory_hits']}, disk={cache_stats['disk_hits']}")
            print(f"      Overall precision: {overall_precision:.2%}")

        return {
            "search_effectiveness": overall_recall,
            "base_recall": base_recall,
            "quantity_score": quantity_score,
            "reference_precision": overall_precision,
            "matched_count": total_matched,
            "total_gt_refs": total_gt,
            "total_pred_refs": total_pred,
            "section_results": section_results,
            "section_stats": section_stats,
            "elapsed_time": elapsed_time,
            "stats": {
                "content_match_count": self.content_match_count,
                "grader_calls": self.grader_calls,
                "grader_total_time": self.grader_total_time,
                "cache_stats": cache_stats,
                "token_usage": {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "total_tokens": self.prompt_tokens + self.completion_tokens,
                },
            },
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.grader_calls = 0
        self.grader_total_time = 0.0
        self.content_match_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
