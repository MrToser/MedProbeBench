import argparse
from pathlib import Path
import json
from typing import Callable, Iterable
import random

def parser_args():
    parser = argparse.ArgumentParser(
        description="HealthBench evaluation (AgentScope + simple-evals rubric)",
    )
    repo_root = Path(__file__).resolve().parents[3]
    default_data = repo_root / "examples" / "agent" / "deep_research_agent" / "datasets" / "2025-05-07-06-14-12_oss_eval.jsonl"
    
    # Agent related
    parser.add_argument(
        "--max_concurrent_agents",
        type=int,
        default=1,
        help="Maximum number of concurrent agents",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name for generating responses (recommended: gpt-4o-mini)",
    )
    parser.add_argument(
        "--result_suffix",
        type=str,
        default="",
        help="Result file suffix",
    )
    
    # Evaluation related
    parser.add_argument(
        "--grader_model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name for rubric grading (recommended: gpt-5)",
    )
    parser.add_argument(
        "--global_judge_model",
        type=str,
        default="gpt-4.1",
        help="OpenAI model name for global evaluation (recommended: gpt-4.1)",
    )
    parser.add_argument(
        "--n_repeat",
        type=int,
        default=1,
        help="Number of evaluation repetitions",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of concurrent workers (HealthBench grading uses sequential calls, recommended: 1)",
    )
    parser.add_argument(
        "--enable_global_eval",
        action="store_true",
    )
    
    
    # Data related
    parser.add_argument(
        "--data_name",
        type=str,
        default=str(
            repo_root
            / "examples"
            / "agent"
            / "deep_research_agent"
            / "results"
        ),
        help="Results save directory",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(default_data),
        help="HealthBench JSONL data path (default uses 5 sample examples)",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="None",
        help="Ground truth data path (JSONL format)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(
            repo_root
            / "examples"
            / "agent"
            / "deep_research_agent"
            / "results"
        ),
        help="Results save directory",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=5,
    )
    # GuideBench related
    parser.add_argument(
        "--weight_task_success_rate",
        type=float,
        default=0.40,
        help="Score weight: claim hit",
    )
    parser.add_argument(
        "--weight_search_effectiveness",
        type=float,
        default=0.15,
        help="Score weight: evidence quality",
    )
    parser.add_argument(
        "--weight_factual_consistency",
        type=float,
        default=0.15,
        help="Score weight: reasoning process",
    )
    parser.add_argument(    
        "--weight_global_eval",
        type=float,
        default=0.30,
        help="Score weight: global evaluation",
        )
    
    
    return parser.parse_args()