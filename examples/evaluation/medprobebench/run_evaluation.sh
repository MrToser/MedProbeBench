#!/bin/bash
# Usage function
usage() {
    echo "Usage: $0 [-m MODEL] [-a] [-n SAMPLES] [-d DATASET] [-h]"
    echo "  -m MODEL    Specify a single model to evaluate"
    echo "  -a          Evaluate all predefined models"
    echo "  -n SAMPLES  Maximum number of samples (default: 1)"
    echo "  -d DATASET  Dataset file path (default: $DATASET_FILE)"
    echo "  -h          Show this help message"
    echo "  -s SUFFIX   Suffix for the results directory"
    exit 1
}

GRADER_MODEL="gpt-4.1"     # Grading model (gpt-4o-mini may score higher)
GLOBAL_JUDEGE_MODEL="gpt-4.1"  # Global grading model

# Predefined models list
ALL_MODELS=(
    # "alibaba/tongyi-deepresearch-30b-a3b"
    # "o4-mini-deep-research"
    # "perplexity/sonar-deep-research"
    # "agentscope"
    # "claude-sonnet-4-20250514"
    # "claude-sonnet-4-20250514-thinking"
    # "gemini-3-flash-preview"
    # "gemini-3-flash-preview-thinking"
    # "gpt-5"
    "gpt-4.1"
    # "gpt-5.2"
    # "grok-4"
    # "Baichuan-M2-Plus"
    # "Baichuan-M3-Plus"
    # "mirothinker-v1.5"
    # "mirothinker-v1.5-pro"
    # "kimi-agent"
)

# Parse command line arguments
MODELS_TO_RUN=()
MAX_SAMPLES=1
DATASET_FILE="./datasets/MedProbeBench.jsonl"
SUFFIX=""
while getopts "m:an:d:s:h" opt; do
    case $opt in
        m)
            MODELS_TO_RUN=("$OPTARG")
            ;;
        a)
            MODELS_TO_RUN=("${ALL_MODELS[@]}")
            ;;
        n)
            MAX_SAMPLES="$OPTARG"
            ;;
        d)
            DATASET_FILE="$OPTARG"
            ;;
        s)
            SUFFIX="$OPTARG"
            ;;
        h)
            usage
            ;;
        *)
            usage
            ;;
    esac
done

# If no arguments provided, show usage
if [ ${#MODELS_TO_RUN[@]} -eq 0 ]; then
    echo "Error: No model specified"
    usage
fi

# Extract dataset name from file path (without extension)
DATASET_NAME=$(basename "$DATASET_FILE" .jsonl)

rm -rf ./prompts/$DATASET_NAME/
python generate_prompts.py -i "$DATASET_FILE" -o ./prompts/$DATASET_NAME/v1/ --max-samples $MAX_SAMPLES
echo "✓ Generated prompts for $DATASET_NAME"

# Function to run model generation based on model type
run_generation() {
    local TEST_MODEL=$1
    echo ">>> Step 1: Running generation for $TEST_MODEL"
    
    case $TEST_MODEL in
        Baichuan-*)
            python baichuan_generator.py --prompt-dir ./prompts/$DATASET_NAME/ --prompt-version v1 \
                -o ./research_output/$DATASET_NAME/$TEST_MODEL/ --max-samples $MAX_SAMPLES --model $TEST_MODEL
            ;;
        claude-*)
            python claude_generator.py --prompt-dir ./prompts/$DATASET_NAME/ --prompt-version v1 \
                -o ./research_output/$DATASET_NAME/$TEST_MODEL/ --max-samples $MAX_SAMPLES --model $TEST_MODEL
            ;;
        gemini-*)
            python gemini_generator.py --prompt-dir ./prompts/$DATASET_NAME/ --prompt-version v1 \
                -o ./research_output/$DATASET_NAME/$TEST_MODEL/ --max-samples $MAX_SAMPLES --model $TEST_MODEL
            ;;
        gpt-*|grok-*)
            python openai_generator.py --prompt-dir ./prompts/$DATASET_NAME/ --prompt-version v1 \
                -o ./research_output/$DATASET_NAME/$TEST_MODEL/ --max-samples $MAX_SAMPLES --model $TEST_MODEL
            ;;
        alibaba/*|perplexity/*|o3-*|o4-*)
            python deep_research_generator.py --prompt-dir ./prompts/$DATASET_NAME/ --prompt-version v1 \
                -o ./research_output/$DATASET_NAME/$TEST_MODEL --max-samples $MAX_SAMPLES --model $TEST_MODEL
            ;;
    esac
}

# Function to run pipeline processing
run_pipeline() {
    local TEST_MODEL=$1
    echo ">>> Step 2: Running pipeline for $TEST_MODEL"
    
    python run_pipeline.py \
        -i ./research_output/$DATASET_NAME/$TEST_MODEL \
        -o ./output/$DATASET_NAME/$TEST_MODEL \
        --model $GRADER_MODEL \
        --output-jsonl ./datasets/$DATASET_NAME/$TEST_MODEL/pred.jsonl \
        --use-llm-enrich \
        --force-extract
}

# Function to run evaluation
run_evaluation() {
    local TEST_MODEL=$1
    echo ">>> Step 3: Running evaluation for $TEST_MODEL"
    
    bash run_medprobebench.sh \
        --data_path "./datasets/$DATASET_NAME/$TEST_MODEL/pred.jsonl" \
        --results_dir "./results/$DATASET_NAME/$TEST_MODEL" \
        --gt_path "$DATASET_FILE" \
        --max_samples "$MAX_SAMPLES" \
        --suffix "$SUFFIX" \
        --global_judge_model $GLOBAL_JUDEGE_MODEL \
        --grader_model $GRADER_MODEL
}

# Main evaluation loop
echo "========================================"
echo "Dataset: $DATASET_NAME"
echo "Starting evaluation for ${#MODELS_TO_RUN[@]} model(s)"
echo "Max samples: $MAX_SAMPLES"
echo "========================================"

# Array to store timing results
declare -A MODEL_TIMES

# Create timing log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TIMING_FILE="./timing_results_${TIMESTAMP}.txt"
echo "Evaluation Timing Report" > "$TIMING_FILE"
echo "Generated at: $(date)" >> "$TIMING_FILE"
echo "Max samples: $MAX_SAMPLES" >> "$TIMING_FILE"
echo "========================================" >> "$TIMING_FILE"
echo "" >> "$TIMING_FILE"

for TEST_MODEL in "${MODELS_TO_RUN[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing model: $TEST_MODEL"
    echo "========================================"
    
    # Record start time
    START_TIME=$(date +%s)
    # run_pipeline "$TEST_MODEL";
    # run_evaluation "$TEST_MODEL"
    
    # Step 1: Generation
    if run_generation "$TEST_MODEL"; then
        # Step 2: Pipeline
        if run_pipeline "$TEST_MODEL"; then
            # Step 3: Evaluation
            run_evaluation "$TEST_MODEL"
        else
            echo "Error: Pipeline failed for $TEST_MODEL"
            echo "Model: $TEST_MODEL - FAILED (Pipeline Error)" >> "$TIMING_FILE"
            continue
        fi
    else
        echo "Error: Generation failed for $TEST_MODEL"
        echo "Model: $TEST_MODEL - FAILED (Generation Error)" >> "$TIMING_FILE"
        continue
    fi
    
    # Calculate elapsed time
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(((ELAPSED % 3600) / 60))
    SECONDS=$((ELAPSED % 60))
    
    # Store timing result
    MODEL_TIMES[$TEST_MODEL]=$(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)
    
    # Write to file
    printf "%-40s %s\n" "$TEST_MODEL" "${MODEL_TIMES[$TEST_MODEL]}" >> "$TIMING_FILE"
    
    echo "✓ Completed: $TEST_MODEL (Time: ${MODEL_TIMES[$TEST_MODEL]})"
    echo ""
done

echo "========================================"
echo "All models processed!"
echo "========================================"
echo ""
echo "Timing Summary:"
echo "----------------------------------------"
# Write summary header to file
echo "" >> "$TIMING_FILE"
echo "========================================" >> "$TIMING_FILE"
echo "Summary:" >> "$TIMING_FILE"
echo "----------------------------------------" >> "$TIMING_FILE"
for model in "${MODELS_TO_RUN[@]}"; do
    if [ -n "${MODEL_TIMES[$model]}" ]; then
        printf "%-40s %s\n" "$model" "${MODEL_TIMES[$model]}"
        printf "%-40s %s\n" "$model" "${MODEL_TIMES[$model]}" >> "$TIMING_FILE"
    fi
done
echo "========================================"
echo "========================================" >> "$TIMING_FILE"
echo ""
echo "Timing results saved to: $TIMING_FILE"