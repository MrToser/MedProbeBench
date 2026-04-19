# -*- coding: utf-8 -*-
"""Claim hit rate evaluation module.

Evaluates whether GT claims are covered by model-generated claims.
Supports async parallel evaluation.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import OpenAI, AsyncOpenAI


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class GTClaim:
    """Ground truth claim."""
    id: str
    claim: str
    reference: list[str]
    type_knowledge: str = ""
    section: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "GTClaim":
        ref = d.get("reference", [])
        if isinstance(ref, str):
            ref = [ref] if ref else []
        return cls(
            id=d.get("id", ""),
            claim=d.get("claim", ""),
            reference=ref,
            type_knowledge=d.get("type_knowledge", ""),
            section=d.get("section", ""),
        )


@dataclass
class PredClaim:
    """Model-generated claim."""
    id: str
    claim: str
    reference: list[str]
    section: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "PredClaim":
        ref = d.get("reference", [])
        if isinstance(ref, str):
            ref = [ref] if ref else []
        return cls(
            id=d.get("id", ""),
            claim=d.get("claim", ""),
            reference=ref,
            section=d.get("section", ""),
        )


# ============================================================================
# LLM Prompt Template (Optimized for fewer tokens)
# ============================================================================

CLAIM_HIT_PROMPT = """Determine if the GT claim is semantically covered by any predicted claim.

GT Claim: {gt_claim}

Predicted Claims:
{pred_claims_json}

Match if: same medical fact, equivalent meaning, or prediction covers GT's core info.

Return JSON only:
{{"is_hit": true/false, "matched_pred_claim_id": "id or empty"}}"""


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


def is_async_client(client: Any) -> bool:
    """Check if client is async."""
    return hasattr(client, 'chat') and hasattr(client.chat.completions, 'create') and \
           asyncio.iscoroutinefunction(getattr(client.chat.completions, 'create', None))


# ============================================================================
# Claim Hit Evaluator
# ============================================================================

class ClaimHitEvaluator:
    """Claim hit rate evaluator (sync and async support)."""

    def __init__(
        self,
        client: "OpenAI | AsyncOpenAI",
        grader_model: str,
        verbose: bool = True,
        similarity_threshold: float = 0.85,
        max_concurrent: int = 10,
    ) -> None:
        self.client = client
        self.grader_model = grader_model
        self.verbose = verbose
        self.similarity_threshold = similarity_threshold
        self.max_concurrent = max_concurrent
        self._semaphore: asyncio.Semaphore | None = None
        
        # Stats
        self.grader_calls = 0
        self.grader_total_time = 0.0
        self.exact_match_count = 0
        self.llm_match_count = 0
        
        # Token usage
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Lazy init semaphore (must be created in event loop)."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    async def _call_grader_async(self, prompt: str) -> str:
        """Async call to grader model."""
        messages = [
            {"role": "system", "content": "You are a medical claim evaluator. Return JSON only."},
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
                        max_tokens=100,  # Reduced from 1000
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

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[，。！？、；：""''【】（）\[\]().,!?;:\"\']', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity."""
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)
        
        if not norm1 or not norm2:
            return 0.0
        if norm1 == norm2:
            return 1.0
        if norm1 in norm2 or norm2 in norm1:
            return 0.9
        
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _find_exact_match(
        self, gt_claim: GTClaim, pred_claims: list[PredClaim]
    ) -> tuple[bool, str, float]:
        """Try exact or high-similarity match (no LLM call)."""
        gt_text = self._normalize_text(gt_claim.claim)
        best_match_id = ""
        best_similarity = 0.0
        
        for pred_claim in pred_claims:
            pred_text = self._normalize_text(pred_claim.claim)
            
            if gt_text == pred_text:
                return True, pred_claim.id, 1.0
            
            similarity = self._compute_text_similarity(gt_claim.claim, pred_claim.claim)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = pred_claim.id
        
        if best_similarity >= self.similarity_threshold:
            return True, best_match_id, best_similarity
        
        return False, best_match_id, best_similarity

    def _validate_matched_id(
        self, 
        llm_matched_id: str, 
        pred_claims: list[PredClaim],
        result_text: str,
        fallback_id: str,
    ) -> str:
        """Validate and fix matched ID from LLM."""
        if not llm_matched_id:
            return ""
        
        valid_ids = [c.id for c in pred_claims]
        
        if llm_matched_id in valid_ids:
            return llm_matched_id
        
        for c in pred_claims:
            if llm_matched_id in c.id or c.id in llm_matched_id:
                return c.id
        
        for valid_id in valid_ids:
            if valid_id in result_text:
                return valid_id
        
        if self.verbose:
            print(f"      ⚠️ Invalid ID '{llm_matched_id}', using fallback: {fallback_id}")
        return fallback_id

    def _build_compact_claims_json(self, pred_claims: list[PredClaim]) -> str:
        """Build compact JSON for pred claims to save tokens."""
        # Only include id and claim, skip section to save tokens
        # Do NOT truncate claims
        compact = [{"id": c.id, "claim": c.claim} for c in pred_claims]
        return json.dumps(compact, ensure_ascii=False, separators=(',', ':'))

    async def evaluate_single_claim_async(
        self, gt_claim: GTClaim, pred_claims: list[PredClaim]
    ) -> dict:
        """Async evaluate single GT claim."""
        if not pred_claims:
            return {
                "gt_claim_id": gt_claim.id,
                "gt_claim": gt_claim.claim,  # No truncation
                "gt_section": gt_claim.section,
                "is_hit": False,
                "matched_pred_claim_id": "",
                "match_method": "no_pred_claims",
                "similarity": 0.0,
            }
        
        # Step 1: Try exact match first (no LLM call) - RESTORED
        exact_match, matched_id, similarity = self._find_exact_match(gt_claim, pred_claims)
        if exact_match:
            self.exact_match_count += 1
            if self.verbose:
                print(f"      ✓ Exact match: GT[{gt_claim.id}] -> Pred[{matched_id}] (similarity: {similarity:.2%})")
            return {
                "gt_claim_id": gt_claim.id,
                "gt_claim": gt_claim.claim,  # No truncation
                "gt_section": gt_claim.section,
                "is_hit": True,
                "matched_pred_claim_id": matched_id,
                "match_method": "exact_or_high_similarity",
                "similarity": similarity,
            }
        
        # Step 2: Use LLM for semantic matching
        pred_claims_json = self._build_compact_claims_json(pred_claims)
        
        # Do NOT truncate GT claim
        prompt = CLAIM_HIT_PROMPT.format(
            gt_claim=gt_claim.claim,
            pred_claims_json=pred_claims_json,
        )
        # print("prompt is", prompt)
        result_text = await self._call_grader_async(prompt)
        # print("result_text is", result_text)
        parsed = parse_json_to_dict(result_text)
        # print("parsed is", parsed)
        is_hit = parsed.get("is_hit", False)
        llm_matched_id = parsed.get("matched_pred_claim_id", "")
        # print("type(is_hit) is", type(is_hit))
        # print("is_hit is", is_hit, "llm_matched_id is", llm_matched_id)
        if is_hit:
            llm_matched_id = self._validate_matched_id(
                llm_matched_id, pred_claims, result_text, matched_id
            )
            self.llm_match_count += 1

        return {
            "gt_claim_id": gt_claim.id,
            "gt_claim": gt_claim.claim,  # No truncation
            "gt_section": gt_claim.section,
            "is_hit": is_hit,
            "matched_pred_claim_id": llm_matched_id if is_hit else "",
            "match_method": "llm_semantic",
            "similarity": similarity,
        }

    async def evaluate(
        self,
        gt_claims: list[GTClaim],
        pred_claims: list[PredClaim],
    ) -> dict:
        """Async evaluate claim hit rate (main entry)."""
        start_time = time.perf_counter()
        
        gt_by_section = self._group_claims_by_section(gt_claims)
        pred_by_section = self._group_claims_by_section(pred_claims)
        
        task_success_rate_results = []
        matched_pred_claims = {}
        section_stats = {}
        
        total_sections = len(gt_by_section)
        for idx, (section, gt_claims_in_section) in enumerate(gt_by_section.items()):
            pred_claims_in_section = pred_by_section.get(section, [])
            
            if self.verbose:
                print(f"    Processing section {idx+1}/{total_sections}: '{section}' "
                      f"(GT: {len(gt_claims_in_section)}, Pred: {len(pred_claims_in_section)})")
            
            tasks = [
                self.evaluate_single_claim_async(gt_claim, pred_claims_in_section)
                for gt_claim in gt_claims_in_section
            ]
            section_results = await asyncio.gather(*tasks)
            
            section_hit_count = 0
            for i, hit_result in enumerate(section_results):
                task_success_rate_results.append(hit_result)
                gt_claim = gt_claims_in_section[i]
                
                if hit_result["is_hit"]:
                    if hit_result["matched_pred_claim_id"]:
                        matched_pred_claims[gt_claim.id] = hit_result["matched_pred_claim_id"]
                    section_hit_count += 1
            
            section_total = len(gt_claims_in_section)
            section_stats[section] = {
                "total_gt_claims": section_total,
                "hit_claims": section_hit_count,
                "hit_rate": section_hit_count / section_total if section_total > 0 else 0.0,
                "pred_claims_count": len(pred_claims_in_section),
            }
            
            if self.verbose:
                print(f"      Section '{section}' done: {section_hit_count}/{section_total} "
                      f"({section_stats[section]['hit_rate']:.1%})")

        hit_count = sum(1 for h in task_success_rate_results if h["is_hit"])
        total_gt_claims = len(gt_claims)
        task_success_rate_rate = hit_count / total_gt_claims if total_gt_claims > 0 else 0.0
        
        elapsed_time = time.perf_counter() - start_time
        
        if self.verbose:
            print(f"  ⏱️  Claim hit evaluation done: {elapsed_time:.2f}s")
            print(f"      Hit rate: {hit_count}/{total_gt_claims} ({task_success_rate_rate:.1%})")
            print(f"      Exact match: {self.exact_match_count}, LLM match: {self.llm_match_count}")

        return {
            "task_success_rate_rate": task_success_rate_rate,
            "hit_count": hit_count,
            "total_gt_claims": total_gt_claims,
            "task_success_rate_results": task_success_rate_results,
            "matched_pred_claims": matched_pred_claims,
            "section_stats": section_stats,
            "elapsed_time": elapsed_time,
            "stats": {
                "exact_match_count": self.exact_match_count,
                "llm_match_count": self.llm_match_count,
                "grader_calls": self.grader_calls,
                "grader_total_time": self.grader_total_time,
                "token_usage": {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "total_tokens": self.prompt_tokens + self.completion_tokens,
                },
            },
        }

    
    def _group_claims_by_section(
        self, claims: list[GTClaim] | list[PredClaim]
    ) -> dict[str, list]:
        """Group claims by section."""
        section_map = {}
        for claim in claims:
            section = claim.section or "Unknown"
            if section not in section_map:
                section_map[section] = []
            section_map[section].append(claim)
        return section_map

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.grader_calls = 0
        self.grader_total_time = 0.0
        self.exact_match_count = 0
        self.llm_match_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
