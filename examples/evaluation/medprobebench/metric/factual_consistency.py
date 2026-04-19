# -*- coding: utf-8 -*-
"""Content consistency evaluation module.

Evaluates whether predicted claims are supported by their referenced URL content.
Supports sync and async evaluation.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from .cache_urls_jina import get_url_cache, URLCacheManager

if TYPE_CHECKING:
    from openai import OpenAI, AsyncOpenAI
    from task_success_rate import PredClaim


# ============================================================================
# LLM Prompt Template (Optimized for fewer tokens)
# ============================================================================

# CONTENT_CONSISTENCY_PROMPT = """Determine if the claim is supported by the reference content.

# Claim: {claim}

# Reference Content:
# {reference_content}

# Support if: content directly states, implies, or provides evidence for the claim.

# Return JSON only:
# {{"is_consistent": true/false}}"""


CONTENT_CONSISTENCY_PROMPT = """Determine whether the claim is meaningfully related to the reference content.

Claim:
{claim}

Reference Content:
{reference_content}

Return true if the claim and the reference content are about the same or closely related specific topic, such that the claim would naturally belong in a discussion based on the reference content, even if the reference content is only an abstract and does not explicitly support the claim.

If the reference content mainly contains **meta-information** (e.g., title, abstract, brief description, or landing page) and lacks substantive details, extract any useful cues such as titles, named objects, or topics.  
In this case, return true if the claim is clearly related to the topic or object indicated by the meta-information.

Return false if the relationship is only very broad or superficial (e.g., they are both about medicine or science in general), or if they are clearly about different subjects.

Do not require direct evidence or factual verification.

Return JSON only:
{{"is_consistent": true/false}}
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
    
    return ""


# ============================================================================
# Content Consistency Evaluator
# ============================================================================

class ContentConsistencyEvaluator:
    """Content consistency evaluator (sync and async support)."""

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
        """
        Initialize evaluator.
        
        Args:
            client: OpenAI client (sync or async)
            grader_model: Grader model name
            verbose: Print verbose info
            fetch_timeout: URL fetch timeout (seconds)
            max_content_length: Max content length
            url_cache: URL cache manager (None uses global singleton)
            max_concurrent: Max concurrent LLM calls
        """
        self.client = client
        self.grader_model = grader_model
        self.verbose = verbose
        self.max_content_length = max_content_length
        self.max_concurrent = max_concurrent
        self._semaphore: asyncio.Semaphore | None = None
        
        # Use shared URL cache
        self._url_cache = url_cache or get_url_cache(
            fetch_timeout=fetch_timeout,
            max_content_length=max_content_length,
            verbose=verbose,
        )
        
        # Stats
        self.grader_calls = 0
        self.grader_total_time = 0.0
        
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
            {"role": "system", "content": "You are a claim verifier. Return JSON only."},
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
                        max_tokens=50,  # Reduced from 500
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
                            max_tokens=50,
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
    # Utility Methods
    # ========================================================================

    def _is_url(self, text: str) -> bool:
        """Check if text is a URL."""
        if not text:
            return False
        try:
            result = urlparse(text.strip())
            return all([result.scheme in ('http', 'https'), result.netloc])
        except:
            return False

    def _collect_urls_from_claim(
        self,
        pred_claim: "PredClaim",
        pred_references: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        """
        Collect all URLs from claim.
        
        Returns:
            tuple: (URL list, non-URL reference list)
        """
        urls = []
        non_url_refs = []
        # print("pred_claim.reference is", pred_claim.reference)
        for ref_id in pred_claim.reference:
            if self._is_url(ref_id):
                urls.append(ref_id)
            elif ref_id in pred_references:
                url = extract_url_from_reference(pred_references[ref_id])
                if url and self._is_url(url):
                    urls.append(url)
                else:
                    non_url_refs.append(ref_id)
            else:
                non_url_refs.append(ref_id)
        
        return urls, non_url_refs

    # ========================================================================
    # Single Claim Evaluation
    # ========================================================================

    async def _evaluate_single_claim_async(
        self,
        pred_claim: "PredClaim",
        url_contents: dict[str, str],
        pred_references: dict[str, str],
    ) -> dict:
        """Async evaluate single predicted claim's content consistency with scoring."""
        if not pred_claim.reference:
            return {
                "pred_claim_id": pred_claim.id,
                "consistency_score": 0.0,
                "reason": "Claim has no references",
                "urls_total": 0,
                "urls_accessible": 0,
                "urls_inaccessible": 0,
                "score_accessible": 0.0,
                "score_inaccessible": 0.0,
            }

        # Collect URLs
        urls, non_url_refs = self._collect_urls_from_claim(pred_claim, pred_references)
        
        if not urls:
            return {
                "pred_claim_id": pred_claim.id,
                "consistency_score": 0.0,
                "reason": f"No accessible URL refs (non-URL refs: {len(non_url_refs)})",
                "urls_total": 0,
                "urls_accessible": 0,
                "urls_inaccessible": 0,
                "non_url_refs": non_url_refs,
                "score_accessible": 0.0,
                "score_inaccessible": 0.0,
            }
        # Separate accessible and inaccessible URLs
        accessible_contents = []
        accessible_urls = []
        inaccessible_urls = []
        
        for url in urls:
            content = url_contents.get(url, "")
            if content and len(content.strip()) > 50:
                content = content[:self.max_content_length]
                accessible_contents.append(content)
                accessible_urls.append(url)
            else:
                inaccessible_urls.append(url)
        
        total_urls = len(urls)
        accessible_count = len(accessible_urls)
        inaccessible_count = len(inaccessible_urls)
        
        score_accessible = 0.0
        is_consistent_llm = False
        
        if accessible_count > 0:
            # Use LLM to judge consistency
            combined_content = "\n---\n".join(accessible_contents)
            prompt = CONTENT_CONSISTENCY_PROMPT.format(
                claim=pred_claim.claim,
                reference_content=combined_content,
            )
            result_text = await self._call_grader_async(prompt)
            parsed = parse_json_to_dict(result_text)
            is_consistent_llm = parsed.get("is_consistent", False)
            
            # If consistent, accessible URLs get full score (50%)
            if is_consistent_llm:
                score_accessible = 1
        
        score_inaccessible = 0
        consistency_score = score_accessible + score_inaccessible

        return {
            "pred_claim_id": pred_claim.id,
            "consistency_score": consistency_score,
            "is_consistent_llm": is_consistent_llm,
            "urls_total": total_urls,
            "urls_accessible": accessible_count,
            "urls_inaccessible": inaccessible_count,
            "accessible_urls": accessible_urls if accessible_urls else None,
            "inaccessible_urls": inaccessible_urls if inaccessible_urls else None,
            "score_accessible": score_accessible,
            "score_inaccessible": score_inaccessible,
        }

    # ========================================================================
    # Main Evaluation Entry
    # ========================================================================

    async def evaluate(
        self,
        pred_claims: list,
        pred_references: dict[str, str],
    ) -> dict:
        """Async evaluate content consistency with scoring (main entry)."""
        start_time = time.perf_counter()
        
        # Collect all needed URLs
        all_urls = set()
        for pred_claim in pred_claims:
            urls, _ = self._collect_urls_from_claim(pred_claim, pred_references)
            all_urls.update(urls)
        
        if self.verbose:
            print(f"    Prefetching {len(all_urls)} URLs (using shared cache)...")
        
        # Async batch fetch URL contents
        url_contents = await self._url_cache.get_batch_async(list(all_urls))
        
        # Parallel evaluate all claims
        tasks = [
            self._evaluate_single_claim_async(pred_claim, url_contents, pred_references)
            for pred_claim in pred_claims
        ]
        
        if self.verbose:
            print(f"    Evaluating {len(pred_claims)} claims for content consistency...")
        
        consistency_results = await asyncio.gather(*tasks)
        
        # Compute stats
        total_score = 0.0
        evaluated_count = 0
        skipped_count = 0
        llm_consistent_count = 0
        
        for result in consistency_results:
            if result.get("urls_total", 0) > 0:
                evaluated_count += 1
                total_score += result["consistency_score"]
                if result.get("is_consistent_llm", False):
                    llm_consistent_count += 1
            else:
                skipped_count += 1

        avg_score = total_score / evaluated_count if evaluated_count > 0 else 0.0
        
        elapsed_time = time.perf_counter() - start_time
        
        # Get cache stats
        cache_stats = self._url_cache.get_stats()
        
        if self.verbose:
            print(f"  ⏱️  Content consistency evaluation done: {elapsed_time:.2f}s")
            print(f"      Total claims: {len(pred_claims)}, with URLs: {evaluated_count}, skipped: {skipped_count}")
            print(f"      Average consistency score: {avg_score:.3f} (range: 0.0-1.0)")
            print(f"      LLM consistent: {llm_consistent_count}/{evaluated_count}")
            print(f"      Cache hits: memory={cache_stats['memory_hits']}, disk={cache_stats['disk_hits']}")

        return {
            "factual_consistency_score": avg_score,
            "total_score": total_score,
            "evaluated_count": evaluated_count,
            "skipped_count": skipped_count,
            "total_claims": len(pred_claims),
            "llm_consistent_count": llm_consistent_count,
            "consistency_results": consistency_results,
            "elapsed_time": elapsed_time,
            "stats": {
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
        self.prompt_tokens = 0
        self.completion_tokens = 0
