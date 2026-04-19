# -*- coding: utf-8 -*-
"""GuideBench evaluation framework - adapted to AgentScope evaluation system.

Evaluation dimensions:
1. Claim coverage (Hit rate): Whether model-generated claims cover key statements in GT
2. Citation correctness:
   - Reference recall: Whether generated guideline citations hit GT claim references
   - Content consistency: Match degree between generated claims and parsed reference content
3. Global scoring (optional): Multi-dimensional evaluation based on global_eval
   - Comprehensiveness (28%)
   - Insight Depth (28%)
   - Accuracy Standards (26%)
   - Readability Utility (18%)

"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI, AsyncOpenAI

from agentscope.evaluate import (
    BenchmarkBase,
    MetricBase,
    MetricResult,
    MetricType,
    SolutionOutput,
    Task,
)

# Import evaluation modules
from .task_success_rate import ClaimHitEvaluator, GTClaim, PredClaim
from .search_effectiveness import ReferenceRecallEvaluator
from .factual_consistency import ContentConsistencyEvaluator
from .cache_urls_jina import get_url_cache
from .guidebench_global_scorer import GlobalEvaluator


# ============================================================================
# Timing Utilities
# ============================================================================

@dataclass
class TimingStats:
    """Evaluation timing statistics."""
    task_success_rate_time: float = 0.0
    search_effectiveness_time: float = 0.0
    factual_consistency_time: float = 0.0
    global_eval_time: float = 0.0
    total_time: float = 0.0
    
    # Detailed statistics
    task_success_rate_calls: int = 0
    search_effectiveness_calls: int = 0
    factual_consistency_calls: int = 0
    global_eval_calls: int = 0
    grader_calls: int = 0
    grader_total_time: float = 0.0
    
    # Token usage statistics
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    
    # Per-module token statistics
    task_success_rate_prompt_tokens: int = 0
    task_success_rate_completion_tokens: int = 0
    search_effectiveness_prompt_tokens: int = 0
    search_effectiveness_completion_tokens: int = 0
    factual_consistency_prompt_tokens: int = 0
    factual_consistency_completion_tokens: int = 0
    global_eval_prompt_tokens: int = 0
    global_eval_completion_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "task_success_rate_time": round(self.task_success_rate_time, 3),
            "search_effectiveness_time": round(self.search_effectiveness_time, 3),
            "factual_consistency_time": round(self.factual_consistency_time, 3),
            "global_eval_time": round(self.global_eval_time, 3),
            "total_time": round(self.total_time, 3),
            "task_success_rate_calls": self.task_success_rate_calls,
            "search_effectiveness_calls": self.search_effectiveness_calls,
            "factual_consistency_calls": self.factual_consistency_calls,
            "global_eval_calls": self.global_eval_calls,
            "grader_calls": self.grader_calls,
            "grader_total_time": round(self.grader_total_time, 3),
            "avg_grader_time": round(self.grader_total_time / self.grader_calls, 3) if self.grader_calls > 0 else 0.0,
            # Token statistics
            "token_usage": {
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
                "by_module": {
                    "task_success_rate": {
                        "prompt_tokens": self.task_success_rate_prompt_tokens,
                        "completion_tokens": self.task_success_rate_completion_tokens,
                        "total": self.task_success_rate_prompt_tokens + self.task_success_rate_completion_tokens,
                    },
                    "search_effectiveness": {
                        "prompt_tokens": self.search_effectiveness_prompt_tokens,
                        "completion_tokens": self.search_effectiveness_completion_tokens,
                        "total": self.search_effectiveness_prompt_tokens + self.search_effectiveness_completion_tokens,
                    },
                    "factual_consistency": {
                        "prompt_tokens": self.factual_consistency_prompt_tokens,
                        "completion_tokens": self.factual_consistency_completion_tokens,
                        "total": self.factual_consistency_prompt_tokens + self.factual_consistency_completion_tokens,
                    },
                    "global_eval": {
                        "prompt_tokens": self.global_eval_prompt_tokens,
                        "completion_tokens": self.global_eval_completion_tokens,
                        "total": self.global_eval_prompt_tokens + self.global_eval_completion_tokens,
                    },
                },
            },
        }


# ============================================================================
# GuideBench Metric - Adapted to AgentScope Evaluation System
# ============================================================================

class GuideBenchMetric(MetricBase):
    """GuideBench comprehensive evaluation metric.
    
    Supports four evaluation dimensions:
    1. task_success_rate: Claim hit rate
    2. search_effectiveness: Reference recall rate
    3. factual_consistency: Content consistency
    4. global_eval: Global multi-dimensional scoring (relative to GT report)
    """

    def __init__(
        self,
        gt_claims: list[GTClaim],
        gt_references: dict[str, Any],
        gt_sample: dict[str, Any] | None = None,  # Complete GT sample (for global scoring)
        grader_model: str = "gpt-4o-mini",
        global_judge_model: str = "gpt-4.1",
        base_url: str | None = None,
        api_key: str | None = None,
        weights: dict[str, float] | None = None,
        verbose_timing: bool = True,
        fetch_timeout: int = 60,
        max_cache_length: int = 30000, # Default 30k characters
        max_concurrent: int = 10,
        enable_global_eval: bool = False,
        global_eval_use_llm: bool = True,
    ) -> None:
        super().__init__(
            name="guidebench_score",
            metric_type=MetricType.NUMERICAL,
            description="GuideBench comprehensive score (Claim hit rate + Reference recall + Content consistency + Global score)",
        )
        self.gt_claims = gt_claims
        self.gt_references = gt_references
        self.gt_sample = gt_sample or {}
        self.grader_model = grader_model
        self.global_judge_model = global_judge_model
        self.base_url = base_url
        self.api_key = api_key
        self.verbose_timing = verbose_timing
        self.fetch_timeout = fetch_timeout
        self.max_cache_length = max_cache_length
        self.max_concurrent = max_concurrent
        self.enable_global_eval = enable_global_eval
        self.global_eval_use_llm = global_eval_use_llm
        
        if enable_global_eval:
            self.weights = weights or {
                "task_success_rate": 0.40,
                "search_effectiveness": 0.15,
                "factual_consistency": 0.15,
                "global_eval": 0.30,
            }
        else:
            self.weights = weights or {
                "task_success_rate": 0.5,
                "search_effectiveness": 0.3,
                "factual_consistency": 0.2,
            }
        
        # Initialize OpenAI client - use AsyncOpenAI to support async calls
        self.async_client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        # Also keep sync client for compatibility
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        
        # Get shared URL cache
        self._url_cache = get_url_cache(
            fetch_timeout=fetch_timeout,
            max_cache_length=max_cache_length,
            verbose=verbose_timing,
        )
        
        # Initialize evaluators - use async_client
        self.task_success_rate_evaluator = ClaimHitEvaluator(
            client=self.async_client,
            grader_model=self.grader_model,
            verbose=self.verbose_timing,
            max_concurrent=self.max_concurrent,
        )
        
        self.search_effectiveness_evaluator = ReferenceRecallEvaluator(
            client=self.async_client,
            grader_model=self.grader_model,
            verbose=self.verbose_timing,
            url_cache=self._url_cache,
            max_concurrent=self.max_concurrent,
        )
        
        self.factual_consistency_evaluator = ContentConsistencyEvaluator(
            client=self.async_client,
            grader_model=self.grader_model,
            verbose=self.verbose_timing,
            url_cache=self._url_cache,
            max_concurrent=self.max_concurrent,
        )
        if "gpt" in self.global_judge_model:
            use_json_mode = True  # Global scoring uses JSON mode
        else:
            use_json_mode = False
        # Global evaluator (optional)
        if enable_global_eval:
            self.global_evaluator = GlobalEvaluator(
                client=self.async_client,
                grader_model=self.global_judge_model,
                use_llm=global_eval_use_llm,
                max_concurrent=self.max_concurrent,
                verbose=self.verbose_timing,
                use_json_mode=use_json_mode,
            )
        else:
            self.global_evaluator = None
        
        # Initialize timing statistics
        self.timing_stats = TimingStats()

    async def __call__(self, solution: SolutionOutput) -> MetricResult:
        """Execute evaluation (async, with full timing)."""
        total_start_time = time.perf_counter()
        
        if not solution.success:
            return MetricResult(
                name=self.name,
                result=0.0,
                metadata={"error": "Generation failed, not scored"},
            )

        # Parse predicted output
        try:
            if isinstance(solution.output, str):
                pred_data = json.loads(solution.output)
            else:
                pred_data = solution.output
            
            pred_claims = [PredClaim.from_dict(c) for c in pred_data.get("claims", [])]
            pred_references = pred_data.get("references", {})
            generated_content = pred_data.get("content", "") or pred_data.get("report", "") or ""
        except Exception as e:
            return MetricResult(
                name=self.name,
                result=0.0,
                metadata={"error": f"Failed to parse predicted output: {e}"},
            )

        if self.verbose_timing:
            print(f"\n📊 Starting evaluation (GT claims: {len(self.gt_claims)}, Pred claims: {len(pred_claims)})")
            print(f"   GT references: {len(self.gt_references)}, Pred references: {len(pred_references)}")
            if self.enable_global_eval:
                print(f"   Global scoring: enabled (LLM: {self.global_eval_use_llm})")

        # 1. Global scoring
        global_eval_score = 0.0
        global_eval_result = None
        
        if self.enable_global_eval and self.global_evaluator and generated_content:
            gt_reference_content = self.gt_sample.get("content", "") or self.gt_sample.get("report", "")
            
            if not gt_reference_content.strip():
                if self.verbose_timing:
                    print("   ⚠️ No reference content found in GT sample, skipping global eval")
            else:
                global_eval_result = await self.global_evaluator.evaluate(
                    sample=self.gt_sample,
                    generated_content=generated_content,
                    reference_content=gt_reference_content,
                )
                global_eval_score = global_eval_result.get("global_score", 0.0)
                
                self.timing_stats.global_eval_time = global_eval_result.get("elapsed_time", 0.0)
                self.timing_stats.global_eval_calls = global_eval_result.get("grader_calls", 0)
                global_token_stats = global_eval_result.get("stats", {}).get("token_usage", {})
                self.timing_stats.global_eval_prompt_tokens = global_token_stats.get("prompt_tokens", 0)
                self.timing_stats.global_eval_completion_tokens = global_token_stats.get("completion_tokens", 0)

        # 2. Claim hit rate
        task_success_rate_result = await self.task_success_rate_evaluator.evaluate(self.gt_claims, pred_claims)
        self.timing_stats.task_success_rate_time = task_success_rate_result["elapsed_time"]
        self.timing_stats.task_success_rate_calls = task_success_rate_result["stats"]["grader_calls"]
        claim_token_stats = task_success_rate_result["stats"].get("token_usage", {})
        self.timing_stats.task_success_rate_prompt_tokens = claim_token_stats.get("prompt_tokens", 0)
        self.timing_stats.task_success_rate_completion_tokens = claim_token_stats.get("completion_tokens", 0)

        # 3. Reference recall
        ref_recall_result = await self.search_effectiveness_evaluator.evaluate(
            self.gt_claims, pred_claims, self.gt_references, pred_references
        )
        self.timing_stats.search_effectiveness_time = ref_recall_result["elapsed_time"]
        self.timing_stats.search_effectiveness_calls = ref_recall_result["stats"]["grader_calls"]
        ref_token_stats = ref_recall_result["stats"].get("token_usage", {})
        self.timing_stats.search_effectiveness_prompt_tokens = ref_token_stats.get("prompt_tokens", 0)
        self.timing_stats.search_effectiveness_completion_tokens = ref_token_stats.get("completion_tokens", 0)

        # 4. Content consistency
        consistency_result = await self.factual_consistency_evaluator.evaluate(pred_claims, pred_references)
        self.timing_stats.factual_consistency_time = consistency_result["elapsed_time"]
        self.timing_stats.factual_consistency_calls = consistency_result["stats"]["grader_calls"]
        consistency_token_stats = consistency_result["stats"].get("token_usage", {})
        self.timing_stats.factual_consistency_prompt_tokens = consistency_token_stats.get("prompt_tokens", 0)
        self.timing_stats.factual_consistency_completion_tokens = consistency_token_stats.get("completion_tokens", 0)

        # Aggregate grader call statistics
        self.timing_stats.grader_calls = (
            task_success_rate_result["stats"]["grader_calls"] +
            ref_recall_result["stats"]["grader_calls"] +
            consistency_result["stats"]["grader_calls"] +
            (global_eval_result["stats"]["grader_calls"] if global_eval_result else 0)
        )
        self.timing_stats.grader_total_time = (
            task_success_rate_result["stats"]["grader_total_time"] +
            ref_recall_result["stats"]["grader_total_time"] +
            consistency_result["stats"]["grader_total_time"] +
            (global_eval_result["stats"]["grader_total_time"] if global_eval_result else 0)
        )
        
        # Aggregate token usage
        self.timing_stats.total_prompt_tokens = (
            self.timing_stats.task_success_rate_prompt_tokens +
            self.timing_stats.search_effectiveness_prompt_tokens +
            self.timing_stats.factual_consistency_prompt_tokens +
            self.timing_stats.global_eval_prompt_tokens
        )
        self.timing_stats.total_completion_tokens = (
            self.timing_stats.task_success_rate_completion_tokens +
            self.timing_stats.search_effectiveness_completion_tokens +
            self.timing_stats.factual_consistency_completion_tokens +
            self.timing_stats.global_eval_completion_tokens
        )
        self.timing_stats.total_tokens = (
            self.timing_stats.total_prompt_tokens +
            self.timing_stats.total_completion_tokens
        )
        
        # 5. Calculate overall score
        overall_score = (
            self.weights.get("task_success_rate", 0) * task_success_rate_result["task_success_rate_rate"] +
            self.weights.get("search_effectiveness", 0) * ref_recall_result["search_effectiveness"] +
            self.weights.get("factual_consistency", 0) * consistency_result["factual_consistency_score"] +
            self.weights.get("global_eval", 0) * global_eval_score
        )

        self.timing_stats.total_time = time.perf_counter() - total_start_time
        url_cache_stats = self._url_cache.get_stats()
        
        if self.verbose_timing:
            print(f"\n⏱️  Total evaluation time: {self.timing_stats.total_time:.2f}s")
            if self.timing_stats.total_time > 0:
                print(f"   - Claim hit rate: {self.timing_stats.task_success_rate_time:.2f}s ({self.timing_stats.task_success_rate_time/self.timing_stats.total_time*100:.1f}%)")
                print(f"   - Reference recall: {self.timing_stats.search_effectiveness_time:.2f}s ({self.timing_stats.search_effectiveness_time/self.timing_stats.total_time*100:.1f}%)")
                print(f"   - Content consistency: {self.timing_stats.factual_consistency_time:.2f}s ({self.timing_stats.factual_consistency_time/self.timing_stats.total_time*100:.1f}%)")
                if self.enable_global_eval:
                    print(f"   - Global scoring: {self.timing_stats.global_eval_time:.2f}s ({self.timing_stats.global_eval_time/self.timing_stats.total_time*100:.1f}%)")
            if self.timing_stats.grader_calls > 0:
                print(f"   - Grader calls: {self.timing_stats.grader_calls} times, avg {self.timing_stats.grader_total_time/self.timing_stats.grader_calls:.2f}s/call")
            print(f"   - URL cache: memory_hits={url_cache_stats['memory_hits']}, disk_hits={url_cache_stats['disk_hits']}, fetched={url_cache_stats['fetch_count']}")
            print(f"\n📊 Token usage statistics:")
            print(f"   - Total: {self.timing_stats.total_tokens:,} tokens (prompt: {self.timing_stats.total_prompt_tokens:,}, completion: {self.timing_stats.total_completion_tokens:,})")
            print(f"   - Claim hit rate: {self.timing_stats.task_success_rate_prompt_tokens + self.timing_stats.task_success_rate_completion_tokens:,} tokens")
            print(f"   - Reference recall: {self.timing_stats.search_effectiveness_prompt_tokens + self.timing_stats.search_effectiveness_completion_tokens:,} tokens")
            print(f"   - Content consistency: {self.timing_stats.factual_consistency_prompt_tokens + self.timing_stats.factual_consistency_completion_tokens:,} tokens")
            if self.enable_global_eval:
                print(f"   - Global scoring: {self.timing_stats.global_eval_prompt_tokens + self.timing_stats.global_eval_completion_tokens:,} tokens")

        # Build metadata — reference sub-result dicts directly to avoid duplication
        metadata = {
            # Scores
            "task_success_rate_rate": task_success_rate_result["task_success_rate_rate"],
            "search_effectiveness": ref_recall_result["search_effectiveness"],
            "factual_consistency_score": consistency_result["factual_consistency_score"],
            "global_score": global_eval_score,
            "weights": self.weights,
            # Claim hit details
            "task_success_rate": {
                "hit_count": task_success_rate_result["hit_count"],
                "total_gt_claims": task_success_rate_result["total_gt_claims"],
                "section_stats": task_success_rate_result["section_stats"],
                "details": task_success_rate_result["task_success_rate_results"],
            },
            # Reference recall details
            "search_effectiveness_detail": {
                "reference_precision": ref_recall_result["reference_precision"],
                "matched_count": ref_recall_result["matched_count"],
                "total_gt_refs": ref_recall_result["total_gt_refs"],
                "total_pred_refs": ref_recall_result["total_pred_refs"],
                "base_recall": ref_recall_result["base_recall"],
                "quantity_score": ref_recall_result["quantity_score"],
                "section_stats": ref_recall_result["section_stats"],
                "section_results": ref_recall_result["section_results"],
            },
            # Content consistency details
            "factual_consistency_detail": {
                "score": consistency_result["factual_consistency_score"],
                "total_score": consistency_result["total_score"],
                "evaluated_claims": consistency_result["evaluated_count"],
                "skipped_claims": consistency_result["skipped_count"],
                "total_claims": len(pred_claims),
                "llm_consistent_count": consistency_result["llm_consistent_count"],
                "details": consistency_result["consistency_results"],
            },
            # Timing and cache
            "timing_stats": self.timing_stats.to_dict(),
            "url_cache_stats": url_cache_stats,
        }
        
        # Global eval details (optional)
        if self.enable_global_eval and global_eval_result:
            metadata["global_eval"] = {
                "score": global_eval_score,
                "dimension_scores": global_eval_result.get("dimension_scores", {}),
                "detailed_scores": global_eval_result.get("detailed_scores", {}),
                "elapsed_time": global_eval_result.get("elapsed_time", 0),
            }
        
        return MetricResult(
            name=self.name,
            result=overall_score,
            metadata=metadata,
        )


# ============================================================================
# GuideBench Benchmark - Adapted to AgentScope Evaluation System
# ============================================================================

class GuideBenchBenchmark(BenchmarkBase):
    """GuideBench dataset - AgentScope BenchmarkBase implementation.
    
    Supports:
    - Loading GT and prediction data
    - Automatic matching of GT and prediction results
    - Configuring evaluation weights and parameters
    - Enabling/disabling global scoring
    """

    def __init__(
        self,
        gt_path: str,
        pred_path: str,
        max_examples: int | None = None,
        grader_model: str = "gpt-4o-mini",
        global_judge_model: str = "gpt-4.1",
        base_url: str | None = None,
        api_key: str | None = None,
        metric_weights: dict[str, float] | None = None,
        verbose_timing: bool = True,
        max_concurrent: int = 10,
        enable_global_eval: bool = False,
        global_eval_use_llm: bool = True,
    ) -> None:
        super().__init__(
            name="GuideBench",
            description="Medical guideline generation evaluation: Claim coverage + Citation correctness + Content consistency + Global score",
        )
        self.gt_path = Path(gt_path).expanduser().resolve()
        self.pred_path = Path(pred_path).expanduser().resolve()
        self.max_examples = max_examples
        self.grader_model = grader_model
        self.global_judge_model = global_judge_model
        self.base_url = base_url
        self.api_key = api_key
        self.metric_weights = metric_weights
        self.verbose_timing = verbose_timing
        self.max_concurrent = max_concurrent
        self.enable_global_eval = enable_global_eval
        self.global_eval_use_llm = global_eval_use_llm
        
        self.gt_data, self.pred_data = self._load_data()
        print(f"len(self.gt_data) {len(self.gt_data)}, len(self.pred_data) {len(self.pred_data)}")

    def _load_jsonl(self, file_path: Path) -> list[dict]:
        """Load JSONL file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        data = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def _load_data(self) -> tuple[dict[str, dict], dict[str, dict]]:
        """Load GT and prediction data."""
        gt_list = self._load_jsonl(self.gt_path)
        pred_list = self._load_jsonl(self.pred_path)
        
        gt_data = {d["id"]: d for d in gt_list}
        pred_data = {d["id"]: d for d in pred_list}
        
        if self.max_examples is not None:
            gt_ids = list(gt_data.keys())[:self.max_examples]
            gt_data = {k: gt_data[k] for k in gt_ids if k in gt_data}
        
        return gt_data, pred_data

    def _match_pred_to_gt(self, gt_id: str) -> dict | None:
        """Find corresponding prediction data by GT ID (supports fuzzy matching)."""
        if gt_id in self.pred_data:
            return self.pred_data[gt_id]
        
        gt_id_lower = gt_id.lower()
        for pred_id, pred_item in self.pred_data.items():
            pred_id_lower = pred_id.lower()
            if gt_id_lower in pred_id_lower or pred_id_lower in gt_id_lower:
                return pred_item
        
        return None

    def _data_to_task(self, gt_id: str, gt_item: dict) -> Task:
        """Convert to evaluation task."""
        gt_claims = [GTClaim.from_dict(c) for c in gt_item.get("claims", [])]
        gt_references = gt_item.get("references", {})
        pred_item = self._match_pred_to_gt(gt_id)
        
        metric = GuideBenchMetric(
            gt_claims=gt_claims,
            gt_references=gt_references,
            gt_sample=gt_item,  # Pass complete GT sample (for global scoring)
            grader_model=self.grader_model,
            global_judge_model=self.global_judge_model,
            base_url=self.base_url,
            api_key=self.api_key,
            weights=self.metric_weights,
            verbose_timing=self.verbose_timing,
            max_concurrent=self.max_concurrent,
            enable_global_eval=self.enable_global_eval,
            global_eval_use_llm=self.global_eval_use_llm,
        )

        return Task(
            id=gt_id,
            input=gt_item.get("prompt", ""),
            ground_truth=gt_item,
            metrics=[metric],
            tags={
                c.type_knowledge: "1"
                for c in gt_claims
                if c.type_knowledge
            },
            metadata={
                "gt_data": gt_item,
                "pred_data": pred_item,
                "has_global_eval": "global_eval" in gt_item,
            },
            search_output=json.dumps(pred_item, ensure_ascii=False) if pred_item else "",
        )

    def __iter__(self):
        for gt_id, gt_item in self.gt_data.items():
            yield self._data_to_task(gt_id, gt_item)

    def __len__(self) -> int:
        return len(self.gt_data)

    def __getitem__(self, index: int) -> Task:
        gt_id = list(self.gt_data.keys())[index]
        return self._data_to_task(gt_id, self.gt_data[gt_id])


# ============================================================================
# Solution Function
# ============================================================================

async def precomputed_solution(
    task: Task,
    pre_hook: Callable,
    model_config: dict,
    verbose: bool = False,
) -> SolutionOutput:
    """Use precomputed results (loaded from report.jsonl)."""
    print(f"Processing task ID: {task.id}")
    try:
        pred_output = task.search_output
        if not pred_output:
            return SolutionOutput(
                success=False,
                output="",
                trajectory=[],
                meta={"error": "No precomputed result found"},
            )
        
        if verbose:
            pred_data = json.loads(pred_output) if isinstance(pred_output, str) else pred_output
            print(f"  Predicted claims count: {len(pred_data.get('claims', []))}")
        
        return SolutionOutput(
            success=True,
            output=pred_output,
            trajectory=[],
        )
    except Exception as e:
        print(f"[Task {task.id}] Error: {e}")
        return SolutionOutput(
            success=False,
            output=str(e),
            trajectory=[],
            meta={"error": str(e)},
        )


def create_model_config(args) -> dict:
    """Build model configuration."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    print(f"base_url: {base_url}")
    print(f"api_key: {api_key[:10]}..." if api_key else "api_key: None")
    return {
        "api_key": api_key,
        "base_url": base_url,
        "grader_model": args.grader_model,
    }