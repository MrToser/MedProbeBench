import os
import uuid
from agentscope.evaluate import (
    FileEvaluatorStorage,
    GeneralEvaluator,
    RayEvaluator,
    SolutionOutput,
    Task,
)

from agentscope import logger

from medprobebench.metric.guidebench import (
    GuideBenchBenchmark,
    precomputed_solution as guide_agent_solution,
    # react_agent_solution as guide_react_agent_solution,
    # native_agent_solution as guide_native_agent_solution,
    create_model_config as guide_create_model_config,
)

async def eval_guidebench(args,pred_path):
    from datetime import datetime
    pred_path = os.path.abspath(pred_path)
    model_config = guide_create_model_config(args)
    metric_weights = {
        "task_success_rate": args.weight_task_success_rate,
        "search_effectiveness": args.weight_search_effectiveness,
        "factual_consistency": args.weight_factual_consistency,
        "global_eval": args.weight_global_eval,
    }
    print("args.enable_global_eval:", args.enable_global_eval)
    
    benchmark = GuideBenchBenchmark(
        gt_path=args.gt_path,
        pred_path=pred_path,
        max_examples=args.max_examples,
        grader_model=args.grader_model,
        global_judge_model=args.global_judge_model,
        base_url=model_config["base_url"],
        api_key=model_config["api_key"],
        metric_weights=metric_weights,
        enable_global_eval=args.enable_global_eval,
        max_concurrent=args.max_concurrent_agents,
    )
    
    print(f"Loaded {len(benchmark)} records from data source: {pred_path}")

    eval_results_dir = args.results_dir + args.result_suffix
    os.makedirs(eval_results_dir, exist_ok=True)
    evaluator = GeneralEvaluator(
        name="HealthBench Evaluation",
        benchmark=benchmark,
        n_repeat=args.n_repeat,
        storage=FileEvaluatorStorage(save_dir=eval_results_dir),
        n_workers=args.n_workers,
    )
    async def solution_wrapper(task: Task, pre_hook) -> SolutionOutput:
        logger.info("Entering solution_wrapper...")
        return await guide_agent_solution(
            task=task,
            pre_hook=pre_hook,
            model_config=model_config,
            verbose=True,
        )
        
    logger.info(f"Grading model={args.grader_model}, output directory={eval_results_dir}")
    await evaluator.run(solution_wrapper)
    # logger.info("Evaluation completed!")
    print("Evaluation completed!")
