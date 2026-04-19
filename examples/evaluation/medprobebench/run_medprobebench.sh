#!/usr/bin/env bash
set -euo pipefail

# ===== medprobebench evaluation script =====
# Used to run evaluation on the medprobebench dataset

# ===== Basic configuration =====
BASE_DIR="./"
DATA_NAME="medprobebench"
# gpt-4.1 gpt-5 gemini-3-flash-preview gpt-4o-mini
# ===== Model configuration =====
# gpt-5 claude-sonnet-4-20250514 gemini-2.5-flash
GRADER_MODEL="gpt-4o-mini"     # Grading model (gpt-4o-mini may score higher)
GLOBAL_JUDEGE_MODEL="gpt-4o-mini"  # Global grading model

# ===== Data path configuration (default values) =====
GT_PATH="$BASE_DIR/datasets/MedProbeBench.jsonl"
DEFAULT_DATA_PATH="$BASE_DIR/datasets/MedProbeBench.jsonl"
DEFAULT_RESULTS_DIR="$BASE_DIR/results/gt/"
SUFFIX=""
# ===== Argument parsing =====
DATA_PATH=""
RESULTS_DIR=$DEFAULT_RESULTS_DIR
MAX_EXAMPLES=1            # Maximum number of evaluation samples

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --results_dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --max_samples)
      MAX_EXAMPLES="$2"
      shift 2
      ;;
    --gt_path)
      GT_PATH="$2"
      shift 2
      ;;
    --suffix)
      SUFFIX="$2"
      shift 2
      ;;
    --global_judge_model)
      GLOBAL_JUDEGE_MODEL="$2"
      shift 2
      ;;
    --grader_model)
      GRADER_MODEL="$2"
      shift 2
      ;;
    *)
      # Ignore other arguments; they will be passed to eval.sh
      shift
      ;;
  esac
done

# Use default value if DATA_PATH is not specified
if [[ -z "$DATA_PATH" ]]; then
  DATA_PATH="$DEFAULT_DATA_PATH"
  echo "⚠️  No --data_path provided, using default: $DATA_PATH"
fi

# Derive RESULTS_DIR from DATA_PATH if not specified
if [[ -z "$RESULTS_DIR" ]]; then
  # Extract directory from DATA_PATH as RESULTS_DIR
  # Example: datasets/model_name/pred.jsonl -> results/model_name/
  DATA_DIR=$(dirname "$DATA_PATH")
  MODEL_DIR=$(basename "$DATA_DIR")
  RESULTS_DIR="$BASE_DIR/results/$MODEL_DIR/"
  echo "⚠️  No --results_dir provided, using inferred: $RESULTS_DIR"
fi

# ===== Runtime parameters =====
MAX_CONCURRENT_AGENTS=100     # Maximum concurrent agent count
N_WORKERS=10                  # Number of concurrent workers

# ===== Evaluation switch =====
ENABLE_GLOBAL_EVAL="--enable_global_eval"  # Enable global scoring (leave empty to disable)

# ===== Weight configuration (optional) =====
WEIGHT_TASK_SUCCESS_RATE=0.4
WEIGHT_SEARCH_EFFECTIVENESS=0.15
WEIGHT_FACTUAL_CONSISTENCY=0.15
WEIGHT_GLOBAL_EVAL=0.3

# ===== Build weight arguments =====
WEIGHT_ARGS=""
if [[ -n "${WEIGHT_TASK_SUCCESS_RATE:-}" ]]; then
  WEIGHT_ARGS="$WEIGHT_ARGS --weight_task_success_rate $WEIGHT_TASK_SUCCESS_RATE"
fi
if [[ -n "${WEIGHT_SEARCH_EFFECTIVENESS:-}" ]]; then
  WEIGHT_ARGS="$WEIGHT_ARGS --weight_search_effectiveness $WEIGHT_SEARCH_EFFECTIVENESS"
fi
if [[ -n "${WEIGHT_FACTUAL_CONSISTENCY:-}" ]]; then
  WEIGHT_ARGS="$WEIGHT_ARGS --weight_factual_consistency $WEIGHT_FACTUAL_CONSISTENCY"
fi
if [[ -n "${WEIGHT_GLOBAL_EVAL:-}" ]]; then
  WEIGHT_ARGS="$WEIGHT_ARGS --weight_global_eval $WEIGHT_GLOBAL_EVAL"
fi

# ===== Switch to script directory =====
cd "$(dirname "${BASH_SOURCE[0]}")"

# ===== Print configuration =====
echo "=========================================="
echo "medprobebench Evaluation"
echo "=========================================="
echo "Grader: $GRADER_MODEL"
echo "GLOBAL_JUDEGE_MODEL: $GLOBAL_JUDEGE_MODEL"
echo "Data Path: $DATA_PATH"
echo "Results Dir: $RESULTS_DIR"
echo "Max Examples: $MAX_EXAMPLES"
echo "=========================================="

# ===== Verify file existence =====
if [[ ! -f "$DATA_PATH" ]]; then
  echo "❌ Error: Data file not found: $DATA_PATH"
  exit 1
fi

if [[ ! -f "$GT_PATH" ]]; then
  echo "❌ Error: Ground truth file not found: $GT_PATH"
  exit 1
fi

# ===== Create result directory =====
mkdir -p "$RESULTS_DIR"

# ===== Run evaluation =====
bash eval.sh \
    --data_path "$DATA_PATH" \
    --results_dir "$RESULTS_DIR" \
    --gt_path "$GT_PATH" \
    --max_examples "$MAX_EXAMPLES" \
    --result_suffix "$SUFFIX" \
    --global_judge_model "$GLOBAL_JUDEGE_MODEL" \
    --grader_model "$GRADER_MODEL" \
    --data_name "$DATA_NAME" \
    --base_dir "$BASE_DIR" \
    --max_concurrent_agents "$MAX_CONCURRENT_AGENTS" \
    --n_workers "$N_WORKERS" \
    $ENABLE_GLOBAL_EVAL \
    $WEIGHT_ARGS

echo "=========================================="
echo "GuideBench evaluation completed!"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="