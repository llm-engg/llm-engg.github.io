#!/usr/bin/env bash
# =============================================================================
# run_vllm_server.sh
# Launch vLLM server for cyankiwi/Qwen3.5-9B-AWQ-4bit with full debug logging
# Shows: continuous batching decisions, KV cache management, scheduler state
# =============================================================================

set -euo pipefail

MODEL="cyankiwi/OmniCoder-9B-AWQ-4bit"
HOST="0.0.0.0"
PORT=8000
MAX_MODEL_LEN=4096
GPU_MEM_UTIL=0.6
BLOCK_SIZE=16
MAX_NUM_SEQS=64          # max concurrent sequences in a batch

echo "============================================="
echo " vLLM Debug Server"
echo " Model : ${MODEL}"
echo " Port  : ${PORT}"
echo " Log   : DEBUG (scheduler + KV cache)"
echo "============================================="

# ---------------------------------------------------------------------------
# Environment variables for granular debug logging
# ---------------------------------------------------------------------------
# VLLM_LOGGING_LEVEL: controls the root vllm logger
export VLLM_LOGGING_LEVEL=DEBUG

# VLLM_TRACE_FUNCTION: trace every function call (very verbose, optional)
# export VLLM_TRACE_FUNCTION=1

# CUDA_LAUNCH_BLOCKING: synchronous CUDA for clearer stack traces (slower)
# export CUDA_LAUNCH_BLOCKING=1

# ---------------------------------------------------------------------------
# Launch the OpenAI-compatible API server
# ---------------------------------------------------------------------------
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype auto \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.6 \
    --block-size "${BLOCK_SIZE}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --enable-prefix-caching \
    2>&1 | tee vllm_server_debug.log

# ---------------------------------------------------------------------------
# Notes on what to observe in the debug output
# ---------------------------------------------------------------------------
# 1. CONTINUOUS BATCHING
#    - Scheduler logs show which requests move between WAITING → RUNNING → SWAPPED
#    - Each iteration logs the batch composition: how many prefill vs decode tokens
#    - Watch for "Schedule" log lines showing num_running, num_waiting, num_swapped
#
# 2. KV CACHE MANAGEMENT (PagedAttention)
#    - Startup: "# GPU blocks: N, # CPU blocks: M" → total KV block budget
#    - Per-request: block allocation/deallocation as requests arrive/finish
#    - Prefix caching: "Prefix cache hit" messages when shared prefixes are reused
#    - Eviction: block eviction logs when memory pressure forces preemption
#
# 3. SCHEDULER DECISIONS
#    - Priority: running (decode) requests first, then waiting (prefill)
#    - Preemption: when GPU blocks run out, requests are swapped to CPU
#    - "Cannot schedule" messages indicate memory pressure
#
# 4. --enforce-eager disables CUDA graphs so every forward pass is traced
#    (remove for production — CUDA graphs give ~10-20% speedup)
#
# Usage:
#   bash run_vllm_server.sh
#   # In another terminal, run the benchmark:
#   python benchmark_transformers_vs_vllm.py
# ---------------------------------------------------------------------------
