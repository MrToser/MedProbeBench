#!/bin/bash

set -u

usage() {
    echo "Usage: $0 [-m MODEL] [-a] [-n SAMPLES] [-d DATASET] [-k] [-h]"
    echo "  -m MODEL    Specify a single model to run generation"
    echo "  -a          Run all predefined models"
    echo "  -n SAMPLES  Maximum number of samples (default: 1)"
    echo "  -d DATASET  Dataset file path (default: ./datasets/MedProbeBench.jsonl)"
    echo "  -k          Keep existing prompts (skip regenerate prompts)"
    echo "  -h          Show this help message"
    exit 1
}

# Keep consistent with run_evaluation.sh
ALL_MODELS=(
    "alibaba/tongyi-deepresearch-30b-a3b"
    "o4-mini-deep-research"
    "perplexity/sonar-deep-research"
    # "agentscope"
    "claude-sonnet-4-20250514"
    "claude-sonnet-4-20250514-thinking"
    "gemini-3-flash-preview"
    "gemini-3-flash-preview-thinking"
    "gpt-5"
    "gpt-4.1"
    "gpt-5.2"
    "grok-4"
    "Baichuan-M2-Plus"
    "Baichuan-M3-Plus"
    # "mirothinker-v1.5"
    # "mirothinker-v1.5-pro"
    # "kimi-agent"
)

MODELS_TO_RUN=()
MAX_SAMPLES=1
DATASET_FILE="./datasets/MedProbeBench.jsonl"
KEEP_PROMPTS=0

while getopts "m:an:d:kh" opt; do
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
        k)
            KEEP_PROMPTS=1
            ;;
        h)
            usage
            ;;
        *)
            usage
            ;;
    esac
done

if [ ${#MODELS_TO_RUN[@]} -eq 0 ]; then
    echo "Error: No model specified"
    usage
fi

DATASET_NAME=$(basename "$DATASET_FILE" .jsonl)
PROMPT_DIR="./prompts/$DATASET_NAME"

if [ "$KEEP_PROMPTS" -eq 0 ]; then
    rm -rf "$PROMPT_DIR/"
    python generate_prompts.py -i "$DATASET_FILE" -o "$PROMPT_DIR/v1/" --max-samples "$MAX_SAMPLES"
    echo "✓ Generated prompts for $DATASET_NAME"
else
    echo "✓ Reusing existing prompts: $PROMPT_DIR/v1/"
fi

run_generation() {
    local TEST_MODEL=$1
    echo ">>> Running generation for $TEST_MODEL"

    case $TEST_MODEL in
        Baichuan-*)
            python baichuan_generator.py --prompt-dir "$PROMPT_DIR/" --prompt-version v1 \
                -o "./research_output/$DATASET_NAME/$TEST_MODEL/" --max-samples "$MAX_SAMPLES" --model "$TEST_MODEL"
            ;;
        claude-*)
            python claude_generator.py --prompt-dir "$PROMPT_DIR/" --prompt-version v1 \
                -o "./research_output/$DATASET_NAME/$TEST_MODEL/" --max-samples "$MAX_SAMPLES" --model "$TEST_MODEL"
            ;;
        gemini-*)
            python gemini_generator.py --prompt-dir "$PROMPT_DIR/" --prompt-version v1 \
                -o "./research_output/$DATASET_NAME/$TEST_MODEL/" --max-samples "$MAX_SAMPLES" --model "$TEST_MODEL"
            ;;
        gpt-*|grok-*)
            python openai_generator.py --prompt-dir "$PROMPT_DIR/" --prompt-version v1 \
                -o "./research_output/$DATASET_NAME/$TEST_MODEL/" --max-samples "$MAX_SAMPLES" --model "$TEST_MODEL"
            ;;
        alibaba/*|perplexity/*|o3-*|o4-*)
            python deep_research_generator.py --prompt-dir "$PROMPT_DIR/" --prompt-version v1 \
                -o "./research_output/$DATASET_NAME/$TEST_MODEL" --max-samples "$MAX_SAMPLES" --model "$TEST_MODEL"
            ;;
        *)
            echo "Error: Unsupported model pattern: $TEST_MODEL"
            return 1
            ;;
    esac
}

echo "========================================"
echo "Dataset: $DATASET_NAME"
echo "Starting GENERATION timing for ${#MODELS_TO_RUN[@]} model(s)"
echo "Max samples: $MAX_SAMPLES"
echo "========================================"

declare -A MODEL_TIMES
declare -A MODEL_SECONDS

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TIMING_FILE="./generation_timing_results_${TIMESTAMP}.txt"
{
    echo "Generation Timing Report"
    echo "Generated at: $(date)"
    echo "Dataset: $DATASET_NAME"
    echo "Max samples: $MAX_SAMPLES"
    echo "========================================"
    echo ""
} > "$TIMING_FILE"

for TEST_MODEL in "${MODELS_TO_RUN[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing model: $TEST_MODEL"
    echo "========================================"

    START_TIME=$(date +%s)
    if run_generation "$TEST_MODEL"; then
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        MODEL_SECONDS[$TEST_MODEL]=$ELAPSED
        HOURS=$((ELAPSED / 3600))
        MINUTES=$(((ELAPSED % 3600) / 60))
        SECONDS=$((ELAPSED % 60))
        MODEL_TIMES[$TEST_MODEL]=$(printf "%02d:%02d:%02d" "$HOURS" "$MINUTES" "$SECONDS")

        printf "%-40s %10s (%6ss)\n" "$TEST_MODEL" "${MODEL_TIMES[$TEST_MODEL]}" "$ELAPSED" >> "$TIMING_FILE"
        echo "✓ Completed: $TEST_MODEL (Generation time: ${MODEL_TIMES[$TEST_MODEL]})"
    else
        echo "Error: Generation failed for $TEST_MODEL"
        echo "Model: $TEST_MODEL - FAILED (Generation Error)" >> "$TIMING_FILE"
    fi
done

echo ""
echo "========================================"
echo "Generation timing summary"
echo "----------------------------------------"
{
    echo ""
    echo "========================================"
    echo "Summary:"
    echo "----------------------------------------"
} >> "$TIMING_FILE"

TOTAL_SECONDS=0
SUCCESS_COUNT=0
for model in "${MODELS_TO_RUN[@]}"; do
    if [ -n "${MODEL_TIMES[$model]:-}" ]; then
        printf "%-40s %10s (%6ss)\n" "$model" "${MODEL_TIMES[$model]}" "${MODEL_SECONDS[$model]}"
        printf "%-40s %10s (%6ss)\n" "$model" "${MODEL_TIMES[$model]}" "${MODEL_SECONDS[$model]}" >> "$TIMING_FILE"
        TOTAL_SECONDS=$((TOTAL_SECONDS + MODEL_SECONDS[$model]))
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    fi
done

if [ "$SUCCESS_COUNT" -gt 0 ]; then
    AVG_SECONDS=$((TOTAL_SECONDS / SUCCESS_COUNT))
    AVG_H=$((AVG_SECONDS / 3600))
    AVG_M=$(((AVG_SECONDS % 3600) / 60))
    AVG_S=$((AVG_SECONDS % 60))
    AVG_FMT=$(printf "%02d:%02d:%02d" "$AVG_H" "$AVG_M" "$AVG_S")
    echo "----------------------------------------"
    echo "Successful models: $SUCCESS_COUNT"
    echo "Average generation time: $AVG_FMT ($AVG_SECONDS s)"
    {
        echo "----------------------------------------"
        echo "Successful models: $SUCCESS_COUNT"
        echo "Average generation time: $AVG_FMT ($AVG_SECONDS s)"
        echo "========================================"
    } >> "$TIMING_FILE"
fi

echo "========================================"
echo "Timing results saved to: $TIMING_FILE"

