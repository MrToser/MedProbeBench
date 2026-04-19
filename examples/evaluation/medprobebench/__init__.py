# -*- coding: utf-8 -*-
"""GuideBench evaluation module.

Provides evaluation capabilities for medical guideline generation:
- Claim hit rate evaluation
- Citation recall evaluation
- Content consistency evaluation
"""

from .metric.task_success_rate import ClaimHitEvaluator, GTClaim, PredClaim
from .metric.search_effectiveness import ReferenceRecallEvaluator
from .metric.factual_consistency import ContentConsistencyEvaluator
from .metric.cache_urls_jina import (
    URLCacheManager,
    get_url_cache,
    fetch_url,
    fetch_urls,
    fetch_url_async,
    fetch_urls_async,
)
from .metric.guidebench import GuideBenchMetric, GuideBenchBenchmark

__all__ = [
    # Data structures
    "GTClaim",
    "PredClaim",
    # Evaluators
    "ClaimHitEvaluator",
    "ReferenceRecallEvaluator",
    "ContentConsistencyEvaluator",
    # URL cache
    "URLCacheManager",
    "get_url_cache",
    "fetch_url",
    "fetch_urls",
    "fetch_url_async",
    "fetch_urls_async",
    # GuideBench main module
    "GuideBenchMetric",
    "GuideBenchBenchmark",
]
