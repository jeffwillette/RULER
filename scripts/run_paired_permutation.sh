#!/bin/bash


source config_models.sh
source config_tasks.sh

BENCHMARK=synthetic
ROOT_DIR="benchmark_root" # the path that stores generated task samples and model predictions.
MODEL_NAME=llama3.1-8b-chat

# smooth interpolation
# DIR_ONE="pred"
# MODEL_ONE="hip_attention-postfix-recompute_dense-window_2048-diff_1-w_64-decode_dense_rerun-smooth_1"

# # SLLM + DELTA
# DIR_ONE="pred_submit"
# MODEL_ONE="hip_attention-postfix-recompute_dense-window_2048-diff_1-w_64-decode_dense"
# 
# # SLLM
# DIR_TWO="pred_submit"
# MODEL_TWO="hip_attention-postfix-recompute_dense-window_2048-diff_0-w_64-decode_dense_JUST_RETURN"




# SLLM + DELTA EMA REBUTTAL
DIR_TWO="pred_submit"
MODEL_TWO="hip_attention-postfix-recompute_dense-window_2048-diff_1-w_64-decode_dense"

# SLLM + DELTA EMA
DIR_ONE="pred"
MODEL_ONE="hip_attention-postfix-recompute_dense-window_2048-diff_1-w_64-decode_dense_rerun-smooth_1-ema_0.05-abgtriton"
# # SLLM + DELTA EMA
# DIR_ONE="pred"
# MODEL_ONE="hip_attention-postfix-recompute_dense-window_2048-diff_1-w_64-decode_dense_rerun-smooth_1-ema_0.95-ematriton"
# # SLLM + DELTA EMA
# DIR_ONE="pred"
# MODEL_ONE="hip_attention-postfix-recompute_dense-window_2048-diff_1-w_64-decode_dense_rerun-smooth_1-ema_0.5-ematriton"

# SLLM + DELTA + SMOOTH
# DIR_ONE="pred"
# MODEL_ONE="hip_attention-postfix-recompute_dense-window_2048-diff_1-w_64-decode_dense_rerun-smooth_1"
#  
# # SLLM + DELTA
# DIR_TWO="pred_submit"
# MODEL_TWO="hip_attention-postfix-recompute_dense-window_2048-diff_1-w_64-decode_dense"

# MInference + DELTA + SMOOTH
# DIR_ONE="pred"
# MODEL_ONE="minference-postfix-delta-window_0-diff_1-w_64-decode_dense-smooth_1"
#  
# # MInference + DELTA
# DIR_TWO="pred_submit"
# MODEL_TWO="minference-postfix-delta-window_0-diff_1-w_64-decode_dense"

# HIP (wrong version)
# DIR_ONE="pred_submit"
# MODEL_ONE="hip_attention-postfix-recompute_dense-window_0-diff_1-w_64-decode_dense"
# 
# # # HIP + DELTA (wrong version)
# DIR_TWO="pred_submit"
# MODEL_TWO="hip_attention-postfix-recompute_dense-window_0-diff_0-w_64-decode_dense_JUST_RETURN"

# MINFERENCE + Delta
# DIR_ONE="pred_submit"
# MODEL_ONE="minference-postfix-delta-window_0-diff_1-w_64-decode_dense"
# 
# # MINFERENCE
# DIR_TWO="pred_submit"
# MODEL_TWO="minference-postfix-plain-window_0-diff_0-w_64-decode_dense"

for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    # if [ $MAX_SEQ_LENGTH -lt 65537 ]; then
    #     echo "continue"
    #     continue  # Skip the iteration when i is 3
    # fi

    echo "::::::: PAIRED PERMUTATION RESULTS FOR ::::::: ${MAX_SEQ_LENGTH} :::::::"

    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    PRED_DIR="${RESULTS_DIR}/pred"

    python eval/evalute_paired_permutation.py \
        --benchmark ${BENCHMARK} \
        --seq-len ${MAX_SEQ_LENGTH} \
        --dir-one ${RESULTS_DIR}/${DIR_ONE} \
        --model-one ${MODEL_ONE} \
        --dir-two ${RESULTS_DIR}/${DIR_TWO} \
        --model-two ${MODEL_TWO}
    
    # RESULTS_DIR2=/data/jeff/hip-ruler-rerun/llama3.1-8b-chat-sglang-18-delta/${BENCHMARK}/${MAX_SEQ_LENGTH}/pred
    # RESULTS_DIR1=/data/jeff/hip-ruler-rerun/llama3.1-8b-chat-sglang-19-delta-smooth/${BENCHMARK}/${MAX_SEQ_LENGTH}/pred
    # RESULTS_DIR1=/data/jeff/hip-ruler-rerun/llama3.1-8b-chat-sglang-18-hip-delta/${BENCHMARK}/${MAX_SEQ_LENGTH}/pred
    # RESULTS_DIR2=/data/jeff/hip-ruler-rerun/llama3.1-8b-chat-sglang-17-hip/${BENCHMARK}/${MAX_SEQ_LENGTH}/pred

    # python eval/evalute_paired_permutation.py \
    #     --benchmark ${BENCHMARK} \
    #     --seq-len ${MAX_SEQ_LENGTH} \
    #     --dir-one ${RESULTS_DIR1} \
    #     --model-one "" \
    #     --dir-two ${RESULTS_DIR2} \
    #     --model-two ""
done

echo "Total time spent on call_api: $total_time seconds"
