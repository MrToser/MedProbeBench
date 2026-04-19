#!/usr/bin/env bash
set -euo pipefail

# ===== Initialize defaults =====
DATA_NAME=""
MODEL_NAME="gpt-4o-mini"
GRADER_MODEL="gpt-4o-mini"
GLOBAL_JUDGE_MODEL="gpt-4.1"
DATA_PATH=""
RESULTS_DIR=""
MAX_CONCURRENT_AGENTS=100
MAX_EXAMPLES=100
BASE_DIR=""
N_WORKERS=1
GT_PATH=""
ENABLE_GLOBAL_EVAL=""
MAX_CONCURRENT=10

# ===== Weight parameters =====
WEIGHT_TASK_SUCCESS_RATE=""
WEIGHT_SEARCH_EFFECTIVENESS=""
WEIGHT_FACTUAL_CONSISTENCY=""
WEIGHT_GLOBAL_EVAL=""
RESULT_SUFFIX=""
# ===== Argument parsing =====
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_name) DATA_NAME="$2"; shift 2;;
    --grader_model) GRADER_MODEL="$2"; shift 2;;
    --global_judge_model) GLOBAL_JUDGE_MODEL="$2"; shift 2;;
    --data_path) DATA_PATH="$2"; shift 2;;
    --results_dir) RESULTS_DIR="$2"; shift 2;;
    --max_concurrent_agents) MAX_CONCURRENT_AGENTS="$2"; shift 2;;
    --max_examples) MAX_EXAMPLES="$2"; shift 2;;
    --base_dir) BASE_DIR="$2"; shift 2;;
    --n_workers) N_WORKERS="$2"; shift 2;;
    --gt_path) GT_PATH="$2"; shift 2;;
    --enable_global_eval) ENABLE_GLOBAL_EVAL="--enable_global_eval"; shift;;
    --max_concurrent) MAX_CONCURRENT="$2"; shift 2;;
    --weight_task_success_rate) WEIGHT_TASK_SUCCESS_RATE="$2"; shift 2;;
    --weight_search_effectiveness) WEIGHT_SEARCH_EFFECTIVENESS="$2"; shift 2;;
    --weight_factual_consistency) WEIGHT_FACTUAL_CONSISTENCY="$2"; shift 2;;
    --weight_global_eval) WEIGHT_GLOBAL_EVAL="$2"; shift 2;;
    --result_suffix) RESULT_SUFFIX="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --data_name NAME              Dataset name (./datasets/medprobebench)"
      echo "  --grader_model MODEL          Claim-level grading model name"
      echo "  --global_judge_model MODEL    Global grading model name"
      echo "  --data_path PATH              Path to prediction jsonl file [required]"
      echo "  --gt_path PATH                Ground-truth data path"
      echo "  --results_dir DIR             Output results directory"
      echo "  --base_dir DIR                Project root directory"
      echo "  --max_concurrent_agents N     Maximum number of concurrent agents (default: 100)"
      echo "  --max_examples N              Maximum number of samples (default: 100)"
      echo "  --n_workers N                 Concurrent workers for agentscope evaluator (default: 1)"
      echo "  --enable_global_eval          Enable global scoring"
      echo "  --weight_task_success_rate W          Weight of claim hit rate"
      echo "  --weight_search_effectiveness W         Weight of reference recall"
      echo "  --weight_factual_consistency W Weight of content consistency"
      echo "  --weight_global_eval W        Weight of global score"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

# ===== Validate key parameters =====
if [[ -z "$DATA_PATH" ]]; then
  echo "ERROR: --data_path is required"
  exit 1
fi

if [[ -z "$BASE_DIR" ]]; then
  # Default to the parent directory of the script location
  BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ -z "$RESULTS_DIR" ]]; then
  RESULTS_DIR="$BASE_DIR/results/${DATA_NAME:-default}"
fi

# ===== Build optional arguments =====
OPTIONAL_ARGS=""

if [[ -n "$GT_PATH" ]]; then
  OPTIONAL_ARGS="$OPTIONAL_ARGS --gt_path $GT_PATH"
fi

if [[ -n "$WEIGHT_TASK_SUCCESS_RATE" ]]; then
  OPTIONAL_ARGS="$OPTIONAL_ARGS --weight_task_success_rate $WEIGHT_TASK_SUCCESS_RATE"
fi

if [[ -n "$WEIGHT_SEARCH_EFFECTIVENESS" ]]; then
  OPTIONAL_ARGS="$OPTIONAL_ARGS --weight_search_effectiveness $WEIGHT_SEARCH_EFFECTIVENESS"
fi

if [[ -n "$WEIGHT_FACTUAL_CONSISTENCY" ]]; then
  OPTIONAL_ARGS="$OPTIONAL_ARGS --weight_factual_consistency $WEIGHT_FACTUAL_CONSISTENCY"
fi

if [[ -n "$WEIGHT_GLOBAL_EVAL" ]]; then
  OPTIONAL_ARGS="$OPTIONAL_ARGS --weight_global_eval $WEIGHT_GLOBAL_EVAL"
fi

# ===== Print configuration =====
echo "=========================================="
echo "Evaluation Configuration:"
echo "=========================================="
echo "  DATA_NAME:    $DATA_NAME"
echo "  MODEL_NAME:   $MODEL_NAME"
echo "  GRADER_MODEL: $GRADER_MODEL"
echo "  GLOBAL_JUDGE_MODEL: ${GLOBAL_JUDGE_MODEL:-N/A}"
echo "  DATA_PATH:    $DATA_PATH"
echo "  GT_PATH:      $GT_PATH"
echo "  RESULTS_DIR:  $RESULTS_DIR"
echo "  BASE_DIR:     $BASE_DIR"
echo "  MAX_EXAMPLES: $MAX_EXAMPLES"
echo "  N_WORKERS:    $N_WORKERS"
echo "  ENABLE_GLOBAL_EVAL: ${ENABLE_GLOBAL_EVAL:-false}"
echo "=========================================="

echo "Starting evaluation..."
# echo "RESULT_SUFFIX is: $RESULT_SUFFIX"
# ===== Run evaluation =====
PYTHONPATH="$BASE_DIR:${PYTHONPATH:-}" \
python "$BASE_DIR/eval.py" \
  --grader_model "$GRADER_MODEL" \
  --result_suffix "$RESULT_SUFFIX" \
  --global_judge_model "$GLOBAL_JUDGE_MODEL" \
  --data_name "$DATA_NAME" \
  --data_path "$DATA_PATH" \
  --results_dir "$RESULTS_DIR" \
  --max_concurrent_agents "$MAX_CONCURRENT_AGENTS" \
  --max_examples "$MAX_EXAMPLES" \
  --n_workers "$N_WORKERS" \
  --max_concurrent "$MAX_CONCURRENT" \
  $OPTIONAL_ARGS \
  $ENABLE_GLOBAL_EVAL

echo "=========================================="
echo "Evaluation completed!"
echo "Results saved in: $RESULTS_DIR"
echo "=========================================="