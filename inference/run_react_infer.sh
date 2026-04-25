#!/bin/bash
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

cd "$( dirname -- "${BASH_SOURCE[0]}" )"

if [ -f ../.env ]; then
    set -a
    source ../.env
    set +a
fi

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

require_env() {
    local name=$1
    if [ -z "${!name:-}" ]; then
        echo "Missing required environment variable: ${name}"
        echo "Set it in inference/.env or export it before running this script."
        exit 1
    fi
}

export TORCHDYNAMO_VERBOSE=${TORCHDYNAMO_VERBOSE:-1}
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}

# NCCL 调试与性能配置
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
export NCCL_IB_TC=${NCCL_IB_TC:-16}
export NCCL_IB_SL=${NCCL_IB_SL:-5}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}
export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-8}
export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-4}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5}

# Gloo backend 配置
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}

# 其他设置
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}

time=$(date +%Y%m%d%H%M%S)
vllm_logs_dir="${VLLM_LOGS_DIR:-./vllm_logs/${time}}"
mkdir -p "${vllm_logs_dir}"

## API Keys for external services
require_env SERPER_KEY_ID
require_env JINA_API_KEYS

##############model servers################
export SUMMARY_MODEL_PATH=${SUMMARY_MODEL_PATH:-Qwen/Qwen3-8B}
export SUMMARY_MODEL_NAME=${SUMMARY_MODEL_NAME:-qwen3-8b}
export SUMMARY_SERVER_PORTS=${SUMMARY_SERVER_PORTS:-7001,7002}
export SUMMARY_CUDA_DEVICES=${SUMMARY_CUDA_DEVICES:-0,0}
export SUMMARY_GPU_MEMORY_UTILIZATION=${SUMMARY_GPU_MEMORY_UTILIZATION:-0.45}

IFS=',' read -r -a summary_ports <<< "${SUMMARY_SERVER_PORTS}"
IFS=',' read -r -a summary_cuda_devices <<< "${SUMMARY_CUDA_DEVICES}"

for i in "${!summary_ports[@]}"; do
    port=${summary_ports[$i]}
    cuda_devices=${summary_cuda_devices[$i]}
    CUDA_VISIBLE_DEVICES=${cuda_devices} vllm serve "${SUMMARY_MODEL_PATH}" \
        --host 0.0.0.0 --port "${port}" \
        --served-model-name "${SUMMARY_MODEL_NAME}" \
        --trust-remote-code --gpu-memory-utilization "${SUMMARY_GPU_MEMORY_UTILIZATION}" \
        > "${vllm_logs_dir}/summary_server_${port}.log" 2>&1 &
done

export LOCAL_MODEL_PATH=${LOCAL_MODEL_PATH:-Alibaba-NLP/Tongyi-DeepResearch-30B-A3B}
export MODEL_PATH=${MODEL_PATH:-alibaba/tongyi-deepresearch-30b-a3b}
export MAIN_SERVER_PORTS=${MAIN_SERVER_PORTS:-6001}
export MAIN_SERVER_PORT=${MAIN_SERVER_PORT:-${MAIN_SERVER_PORTS%%,*}}
MAIN_CUDA_DEVICES=${MAIN_CUDA_DEVICES:-1,2}
MAIN_TENSOR_PARALLEL_SIZE=${MAIN_TENSOR_PARALLEL_SIZE:-$(awk -F',' '{print NF}' <<< "${MAIN_CUDA_DEVICES}")}
MAIN_GPU_MEMORY_UTILIZATION=${MAIN_GPU_MEMORY_UTILIZATION:-0.85}

CUDA_VISIBLE_DEVICES=${MAIN_CUDA_DEVICES} vllm serve "${LOCAL_MODEL_PATH}" \
    --host 0.0.0.0 --port "${MAIN_SERVER_PORT}" \
    --served-model-name "${MODEL_PATH}" \
    --trust-remote-code \
    --tensor-parallel-size "${MAIN_TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${MAIN_GPU_MEMORY_UTILIZATION}" \
    > "${vllm_logs_dir}/main_server_${MAIN_SERVER_PORT}.log" 2>&1 &


##############hyperparams################
export TOKENIZER_PATH=${TOKENIZER_PATH:-${LOCAL_MODEL_PATH}}
export ROLLOUT_COUNT=${ROLLOUT_COUNT:-1}
export TEMPERATURE=${TEMPERATURE:-0.6}
export MAX_WORKERS=${MAX_WORKERS:-16}

export JUDGE_ENGINE=${JUDGE_ENGINE:-deepseekchat}

## OpenAI API configuration (optional, for summary model)
export SUMMARY_SERVER_PORT=${SUMMARY_SERVER_PORT:-${summary_ports[0]}}
export API_BASE=${API_BASE:-http://127.0.0.1:${SUMMARY_SERVER_PORT}/v1}
export API_KEY=${API_KEY:-EMPTY}


export TORCH_COMPILE_CACHE_DIR=${TORCH_COMPILE_CACHE_DIR:-./cache}

timeout=${VLLM_STARTUP_TIMEOUT:-6000}
start_time=$(date +%s)

wait_for_vllm() {
    local port=$1
    local name=$2
    echo "Waiting for ${name} vLLM server on port ${port}..."
    while true; do
        if curl -fsS "http://127.0.0.1:${port}/v1/models" > /dev/null 2>&1; then
            echo "${name} vLLM server on port ${port} is ready."
            break
        fi
        if (( $(date +%s) - start_time > timeout )); then
            echo "Timeout waiting for ${name} vLLM server on port ${port}."
            exit 1
        fi
        sleep 5
    done
}

#####################################
### 3. start infer               ####
#####################################

echo "==== start WebExplorer evaluation... ===="

for port in "${summary_ports[@]}"; do
    wait_for_vllm "${port}" "summary"
done
wait_for_vllm "${MAIN_SERVER_PORT}" "main"

export DATASET=${DATASET:-researchqa.jsonl}
export OUTPUT_PATH=${OUTPUT_PATH:-researchqa_output_2}
python -u run_search_visit.py --dataset "$DATASET" --output "$OUTPUT_PATH" --max_workers $MAX_WORKERS --model $MODEL_PATH --temperature $TEMPERATURE --total_splits ${WORLD_SIZE:-1} --worker_split $((${RANK:-0} + 1)) --roll_out_count $ROLLOUT_COUNT