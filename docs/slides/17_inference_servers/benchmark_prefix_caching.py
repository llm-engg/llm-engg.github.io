"""
benchmark_prefix_caching.py
============================
Demonstrates and benchmarks vLLM prefix caching across realistic scenarios.

Requires a vLLM server running with --enable-prefix-caching (see run_vllm_server.sh).

Experiments:
  1. System Prompt Caching  — same long system prompt, different user queries
  2. Multi-turn Conversation — growing chat history reused each turn
  3. RAG Context Caching     — same document, different questions
  4. Cold vs Warm Batch      — repeat identical batch to show cache hit speedup

Prerequisites:
    pip install openai rich

Usage:
    python benchmark_prefix_caching.py
    python benchmark_prefix_caching.py --experiment system_prompt
    python benchmark_prefix_caching.py --experiment multi_turn
    python benchmark_prefix_caching.py --experiment rag
    python benchmark_prefix_caching.py --experiment cold_vs_warm
    python benchmark_prefix_caching.py --max-tokens 128 --runs 5
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_ID = "cyankiwi/OmniCoder-9B-AWQ-4bit"
DEFAULT_MAX_TOKENS = 100
DEFAULT_RUNS = 3  # repeat each experiment N times for stable measurements


# ---------------------------------------------------------------------------
# Shared prefixes for experiments
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_LONG = """\
You are an expert software engineering assistant with deep knowledge of \
distributed systems, databases, machine learning, and cloud infrastructure. \
You follow these principles strictly:

1. Always provide working, production-quality code examples.
2. Consider edge cases, error handling, and performance implications.
3. When explaining concepts, use analogies that connect to the user's domain.
4. If a question is ambiguous, state your assumptions before answering.
5. For architecture decisions, present tradeoffs rather than single answers.
6. Use Python type hints and docstrings in all code examples.
7. Reference specific versions of libraries when relevant.
8. Consider security implications of any code you write.
9. Prefer standard library solutions over third-party when possible.
10. Break complex problems into smaller, testable components.

Your responses should be concise but thorough. Always explain WHY, not just HOW. \
When you write code, include brief inline comments for non-obvious logic. \
If the user's request could be interpreted multiple ways, address the most \
likely interpretation first, then mention alternatives.

You have access to the following tools and can use them when helpful:
- execute_code: Run Python code in a sandboxed environment
- search_docs: Search documentation for libraries and frameworks
- query_database: Run read-only SQL queries against the project database
- list_files: Browse the project directory structure
- read_file: Read file contents from the project

Current project context: A distributed task queue system built with Python, \
Redis, and PostgreSQL. The system processes ~50K tasks/hour across 20 worker \
nodes. Recent focus areas include improving task routing, reducing p99 latency, \
and adding observability. The codebase uses FastAPI for the control plane, \
SQLAlchemy for ORM, and Celery for task execution.

Important constraints:
- Python 3.11+ required
- All database operations must be async
- Task processing must be idempotent
- Maximum task payload size: 1MB
- Worker nodes run on k8s with 4 CPU / 8GB RAM each
"""

RAG_DOCUMENT = """\
Context document for answering questions:

# PagedAttention: Virtual Memory for Large Language Models

## Abstract
Large language models (LLMs) require significant GPU memory for storing key-value
(KV) cache during inference. Existing systems waste 60-80% of KV cache memory due
to fragmentation and over-reservation. We propose PagedAttention, an attention
algorithm inspired by virtual memory and paging in operating systems. PagedAttention
divides the KV cache into fixed-size blocks that can be stored non-contiguously in
physical GPU memory. This allows flexible memory management: blocks are allocated
on demand, shared across sequences (for parallel sampling and beam search), and
freed immediately when no longer needed.

## 1. Introduction
The KV cache stores previously computed key and value tensors so they don't need
to be recomputed at each generation step. For a model like LLaMA-13B with a 2048
context length, a single sequence's KV cache requires ~1.6GB of GPU memory. When
serving multiple concurrent requests, KV cache memory becomes the primary bottleneck.

Current systems allocate KV cache as contiguous tensors:
- Pre-allocation wastes memory (must reserve for max sequence length)
- Internal fragmentation (allocated blocks partially used)
- External fragmentation (free memory is non-contiguous, unusable)
- No sharing between sequences with common prefixes

## 2. PagedAttention Algorithm
### 2.1 Block Structure
Each block stores KV vectors for a fixed number of tokens (block_size, typically 16).
Blocks are analogous to pages in OS virtual memory. A block table maps logical blocks
(sequence position) to physical blocks (GPU memory location).

### 2.2 Memory Management
The block manager tracks free/used blocks using a free list. On each decode step:
1. Check if current block has space → append token KV to existing block
2. If full → allocate new block from free list
3. If free list empty → trigger preemption (swap to CPU or recompute)

### 2.3 Copy-on-Write for Shared Prefixes
When multiple sequences share a prefix (e.g., beam search candidates), their block
tables point to the same physical blocks. When a sequence diverges, the shared block
is copied (copy-on-write), and only the divergent block is modified.

## 3. Implementation in vLLM
vLLM implements PagedAttention as its core memory management system:
- Block size: configurable (default 16 tokens)
- Supports GPU-to-CPU block swapping for preemption
- Prefix caching: hash-based lookup of previously computed KV blocks
- Integration with continuous batching scheduler

## 4. Results
Compared to FasterTransformer and Orca:
- 2-4x throughput improvement at same latency
- Near-zero memory waste (< 4% fragmentation)
- Enables serving 2-3x more concurrent sequences
- Effective for long sequences (4K-32K context)

## 5. Prefix Caching Extension
vLLM extends PagedAttention with automatic prefix caching:
- Each block of tokens is hashed (content-addressable)
- On new request, hash each prefix block and check cache
- Cache hit: reuse existing physical KV block (zero compute)
- Cache miss: compute and store for future reuse
- LRU eviction when cache is full
- Reduces TTFT by 5-20x for shared prefixes
"""

USER_QUERIES_SYSTEM_PROMPT = [
    "How can I reduce p99 latency in our task queue? Focus on Redis-specific optimizations.",
    "Write a Python decorator that makes any Celery task idempotent using PostgreSQL advisory locks.",
    "What's the best strategy for implementing priority queues in our Celery + Redis setup?",
    "Design an observability dashboard for our task queue. What metrics should we track?",
    "How should we handle poison pill tasks that consistently fail and block workers?",
    "Write an async SQLAlchemy query that finds tasks stuck in 'processing' for over 10 minutes.",
    "What's the best way to implement task routing based on worker node capabilities?",
    "How can we implement graceful shutdown for Celery workers on Kubernetes?",
    "Design a dead letter queue strategy for our system. Include retry backoff logic.",
    "Write a FastAPI endpoint that returns real-time task queue depth per priority level.",
    "How should we handle database connection pooling with async SQLAlchemy across 20 workers?",
    "What's the most efficient way to batch-insert task results into PostgreSQL?",
    "Design a circuit breaker for external API calls within our task workers.",
    "How can we implement task deduplication to prevent processing the same event twice?",
    "Write a health check endpoint that verifies Redis, PostgreSQL, and worker connectivity.",
]

USER_QUERIES_RAG = [
    "What is PagedAttention and what problem does it solve?",
    "How does the block structure work in PagedAttention? What is the typical block size?",
    "Explain the copy-on-write mechanism for shared prefixes.",
    "What happens when the free list is empty during decoding?",
    "How does prefix caching work in vLLM? What hashing strategy is used?",
    "What throughput improvements does PagedAttention achieve over FasterTransformer?",
    "How does PagedAttention reduce memory fragmentation compared to contiguous allocation?",
    "Explain how block tables map logical blocks to physical blocks.",
    "What is the memory requirement for a single sequence's KV cache in LLaMA-13B?",
    "How does vLLM's prefix caching reduce TTFT? What speedup range is typical?",
]

MULTI_TURN_CONVERSATION = [
    ("What is continuous batching and how does it differ from static batching?", None),
    ("Can you show me a code example of how a scheduler might implement this?", None),
    ("How does this interact with PagedAttention's memory management?", None),
    ("What happens when a new request arrives mid-batch during decoding?", None),
    ("How would you monitor batch utilization in production? What metrics matter?", None),
    ("Show me how to implement a simple priority-aware continuous batching scheduler in Python.", None),
    ("What are the failure modes? How do you handle OOM during continuous batching?", None),
    ("Compare the scheduling strategies of vLLM, TensorRT-LLM, and SGLang.", None),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RequestMetrics:
    prompt_label: str
    ttft_s: float
    total_latency_s: float
    output_tokens: int
    is_cache_warm: bool


@dataclass
class ExperimentResult:
    name: str
    description: str
    metrics: list[RequestMetrics]

    def cold_metrics(self) -> list[RequestMetrics]:
        return [m for m in self.metrics if not m.is_cache_warm]

    def warm_metrics(self) -> list[RequestMetrics]:
        return [m for m in self.metrics if m.is_cache_warm]

    def summary(self) -> dict:
        cold = self.cold_metrics()
        warm = self.warm_metrics()

        def stats(items: list[RequestMetrics], field: str) -> dict:
            if not items:
                return {}
            vals = [getattr(m, field) for m in items]
            return {
                "count": len(vals),
                "mean": statistics.mean(vals),
                "median": statistics.median(vals),
                "stdev": statistics.stdev(vals) if len(vals) > 1 else 0,
                "min": min(vals),
                "max": max(vals),
            }

        result = {"experiment": self.name, "description": self.description}
        if cold:
            result["cold_ttft"] = stats(cold, "ttft_s")
            result["cold_latency"] = stats(cold, "total_latency_s")
        if warm:
            result["warm_ttft"] = stats(warm, "ttft_s")
            result["warm_latency"] = stats(warm, "total_latency_s")
        if cold and warm:
            cold_ttft = statistics.mean(m.ttft_s for m in cold)
            warm_ttft = statistics.mean(m.ttft_s for m in warm)
            if warm_ttft > 0:
                result["ttft_speedup"] = cold_ttft / warm_ttft
        return result


# ---------------------------------------------------------------------------
# Core request function
# ---------------------------------------------------------------------------

async def send_request(
    client: AsyncOpenAI,
    prompt: str,
    max_tokens: int,
    label: str = "",
    is_cache_warm: bool = False,
) -> RequestMetrics:
    """Send a streaming completion request and measure TTFT + total latency."""
    t0 = time.perf_counter()
    ttft = None
    token_count = 0

    stream = await client.completions.create(
        model=MODEL_ID,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        stream=True,
    )

    async for chunk in stream:
        if ttft is None:
            ttft = time.perf_counter() - t0
        if chunk.choices and chunk.choices[0].text:
            token_count += 1  # approximate: 1 chunk ≈ 1 token in streaming

    total = time.perf_counter() - t0

    return RequestMetrics(
        prompt_label=label,
        ttft_s=ttft if ttft is not None else total,
        total_latency_s=total,
        output_tokens=token_count,
        is_cache_warm=is_cache_warm,
    )


async def send_request_sequential(
    client: AsyncOpenAI,
    prompt: str,
    max_tokens: int,
    label: str = "",
    is_cache_warm: bool = False,
) -> RequestMetrics:
    """Same as send_request but designed for sequential experiments where we
    want to isolate cache effects (no concurrent request interference)."""
    return await send_request(client, prompt, max_tokens, label, is_cache_warm)


# ---------------------------------------------------------------------------
# Experiment 1: System Prompt Caching
# ---------------------------------------------------------------------------

async def experiment_system_prompt(
    client: AsyncOpenAI, max_tokens: int, runs: int
) -> ExperimentResult:
    """
    Send requests with the SAME long system prompt but different user queries.
    First request (cold): full prefill of system prompt.
    Subsequent requests (warm): system prompt KV blocks served from cache.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: System Prompt Prefix Caching")
    print("  Long system prompt (~600 tokens) + varying user queries")
    print("=" * 70)

    metrics: list[RequestMetrics] = []
    queries = USER_QUERIES_SYSTEM_PROMPT[:10]

    for run_idx in range(runs):
        print(f"\n  Run {run_idx + 1}/{runs}")
        for i, query in enumerate(queries):
            prompt = f"{SYSTEM_PROMPT_LONG}\n\nUser: {query}\nAssistant:"
            is_warm = (run_idx > 0) or (i > 0)  # first request of first run is cold
            label = f"run{run_idx+1}_q{i+1}"

            m = await send_request_sequential(
                client, prompt, max_tokens, label, is_cache_warm=is_warm
            )
            marker = "WARM" if is_warm else "COLD"
            print(f"    [{marker}] q{i+1:2d} TTFT={m.ttft_s:.3f}s  total={m.total_latency_s:.3f}s")
            metrics.append(m)

    return ExperimentResult(
        name="system_prompt_caching",
        description="Same system prompt (~600 tokens) + different user queries",
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Experiment 2: Multi-turn Conversation Caching
# ---------------------------------------------------------------------------

async def experiment_multi_turn(
    client: AsyncOpenAI, max_tokens: int, runs: int
) -> ExperimentResult:
    """
    Simulate a multi-turn conversation where history grows each turn.
    With prefix caching, TTFT should stay roughly constant as only the new
    turn needs prefill — all previous turns hit the cache.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Multi-turn Conversation Caching")
    print("  Growing chat history — TTFT should stay ~constant with caching")
    print("=" * 70)

    metrics: list[RequestMetrics] = []

    for run_idx in range(runs):
        print(f"\n  Run {run_idx + 1}/{runs}")
        conversation = f"{SYSTEM_PROMPT_LONG}\n\n"
        simulated_responses = []

        for turn_idx, (user_msg, _) in enumerate(MULTI_TURN_CONVERSATION):
            # Build prompt with full history
            conversation += f"User: {user_msg}\nAssistant:"
            is_warm = (run_idx > 0) or (turn_idx > 0)
            label = f"run{run_idx+1}_turn{turn_idx+1}"

            m = await send_request_sequential(
                client, conversation, max_tokens, label, is_cache_warm=is_warm
            )

            # Simulate assistant response in history for next turn
            fake_response = f" [Response to turn {turn_idx + 1}, ~{m.output_tokens} tokens]"
            conversation += fake_response + "\n\n"

            prompt_len_approx = len(conversation.split())
            marker = "WARM" if is_warm else "COLD"
            print(
                f"    [{marker}] turn {turn_idx+1:2d} | "
                f"~{prompt_len_approx:4d} words in prompt | "
                f"TTFT={m.ttft_s:.3f}s  total={m.total_latency_s:.3f}s"
            )
            metrics.append(m)

    return ExperimentResult(
        name="multi_turn_conversation",
        description="Growing conversation history — measures TTFT stability with caching",
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Experiment 3: RAG Context Caching
# ---------------------------------------------------------------------------

async def experiment_rag(
    client: AsyncOpenAI, max_tokens: int, runs: int
) -> ExperimentResult:
    """
    Same RAG document context + different questions.
    After the first request caches the document's KV blocks, subsequent
    questions only need to prefill the question tokens.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: RAG Document Context Caching")
    print("  Same document (~800 tokens) + different questions")
    print("=" * 70)

    metrics: list[RequestMetrics] = []

    for run_idx in range(runs):
        print(f"\n  Run {run_idx + 1}/{runs}")
        for i, question in enumerate(USER_QUERIES_RAG):
            prompt = (
                f"Based on the following document, answer the question.\n\n"
                f"---\n{RAG_DOCUMENT}\n---\n\n"
                f"Question: {question}\nAnswer:"
            )
            is_warm = (run_idx > 0) or (i > 0)
            label = f"run{run_idx+1}_q{i+1}"

            m = await send_request_sequential(
                client, prompt, max_tokens, label, is_cache_warm=is_warm
            )
            marker = "WARM" if is_warm else "COLD"
            print(f"    [{marker}] q{i+1:2d} TTFT={m.ttft_s:.3f}s  total={m.total_latency_s:.3f}s")
            metrics.append(m)

    return ExperimentResult(
        name="rag_context_caching",
        description="Same RAG document (~800 tokens) + different questions",
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Experiment 4: Cold vs Warm Batch
# ---------------------------------------------------------------------------

async def experiment_cold_vs_warm(
    client: AsyncOpenAI, max_tokens: int, runs: int
) -> ExperimentResult:
    """
    Send the exact same batch of requests twice:
      - Batch 1 (cold): no prefix cache populated
      - Batch 2 (warm): identical prompts → full cache hits

    Sends requests concurrently within each batch to show aggregate speedup.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4: Cold vs Warm Batch (concurrent)")
    print("  Same 10 prompts sent twice — second batch should be faster")
    print("=" * 70)

    prompts = []
    for i, query in enumerate(USER_QUERIES_SYSTEM_PROMPT[:10]):
        prompts.append(f"{SYSTEM_PROMPT_LONG}\n\nUser: {query}\nAssistant:")

    metrics: list[RequestMetrics] = []
    semaphore = asyncio.Semaphore(8)

    async def bounded_request(idx, prompt, is_warm, batch_label):
        async with semaphore:
            return await send_request(
                client, prompt, max_tokens,
                label=f"{batch_label}_q{idx+1}",
                is_cache_warm=is_warm,
            )

    for run_idx in range(runs):
        print(f"\n  Run {run_idx + 1}/{runs}")

        # Cold batch
        print("    Batch 1 (COLD)...")
        t0 = time.perf_counter()
        cold_tasks = [bounded_request(i, p, False, f"run{run_idx+1}_cold") for i, p in enumerate(prompts)]
        cold_results = await asyncio.gather(*cold_tasks)
        cold_wall = time.perf_counter() - t0
        metrics.extend(cold_results)

        cold_ttfts = [m.ttft_s for m in cold_results]
        print(f"      Wall time: {cold_wall:.2f}s | "
              f"Avg TTFT: {statistics.mean(cold_ttfts):.3f}s | "
              f"Min TTFT: {min(cold_ttfts):.3f}s | "
              f"Max TTFT: {max(cold_ttfts):.3f}s")

        # Brief pause to ensure cache is settled
        await asyncio.sleep(0.5)

        # Warm batch (identical prompts)
        print("    Batch 2 (WARM)...")
        t0 = time.perf_counter()
        warm_tasks = [bounded_request(i, p, True, f"run{run_idx+1}_warm") for i, p in enumerate(prompts)]
        warm_results = await asyncio.gather(*warm_tasks)
        warm_wall = time.perf_counter() - t0
        metrics.extend(warm_results)

        warm_ttfts = [m.ttft_s for m in warm_results]
        print(f"      Wall time: {warm_wall:.2f}s | "
              f"Avg TTFT: {statistics.mean(warm_ttfts):.3f}s | "
              f"Min TTFT: {min(warm_ttfts):.3f}s | "
              f"Max TTFT: {max(warm_ttfts):.3f}s")

        if statistics.mean(warm_ttfts) > 0:
            speedup = statistics.mean(cold_ttfts) / statistics.mean(warm_ttfts)
            print(f"      TTFT speedup: {speedup:.1f}x")

    return ExperimentResult(
        name="cold_vs_warm_batch",
        description="Identical batch sent twice — concurrent, measures aggregate cache benefit",
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(results: list[ExperimentResult]):
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    for r in results:
        s = r.summary()
        print(f"\n  --- {s['experiment']} ---")
        print(f"  {s['description']}")

        if "cold_ttft" in s:
            ct = s["cold_ttft"]
            print(f"  Cold TTFT : mean={ct['mean']:.3f}s  median={ct['median']:.3f}s  "
                  f"min={ct['min']:.3f}s  max={ct['max']:.3f}s  (n={ct['count']})")

        if "warm_ttft" in s:
            wt = s["warm_ttft"]
            print(f"  Warm TTFT : mean={wt['mean']:.3f}s  median={wt['median']:.3f}s  "
                  f"min={wt['min']:.3f}s  max={wt['max']:.3f}s  (n={wt['count']})")

        if "ttft_speedup" in s:
            print(f"  >> TTFT Speedup (cold/warm): {s['ttft_speedup']:.1f}x")

        if "cold_latency" in s:
            cl = s["cold_latency"]
            print(f"  Cold total: mean={cl['mean']:.3f}s  median={cl['median']:.3f}s")

        if "warm_latency" in s:
            wl = s["warm_latency"]
            print(f"  Warm total: mean={wl['mean']:.3f}s  median={wl['median']:.3f}s")

    print()


def save_results(results: list[ExperimentResult], output_path: str):
    data = {r.name: r.summary() for r in results}

    # Also include raw metrics for further analysis
    for r in results:
        data[r.name]["raw_metrics"] = [
            {
                "label": m.prompt_label,
                "ttft_s": m.ttft_s,
                "total_latency_s": m.total_latency_s,
                "output_tokens": m.output_tokens,
                "is_cache_warm": m.is_cache_warm,
            }
            for m in r.metrics
        ]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "system_prompt": experiment_system_prompt,
    "multi_turn": experiment_multi_turn,
    "rag": experiment_rag,
    "cold_vs_warm": experiment_cold_vs_warm,
}


async def async_main(args):
    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")

    # Verify server is reachable
    try:
        models = await client.models.list()
        available = [m.id for m in models.data]
        print(f"  Server OK — models: {available}")
    except Exception as e:
        print(f"\n  ERROR: Cannot reach vLLM at {VLLM_BASE_URL}")
        print(f"  {e}")
        print(f"  Start server first: bash run_vllm_server.sh")
        sys.exit(1)

    # Determine which experiments to run
    if args.experiment == "all":
        to_run = list(EXPERIMENTS.keys())
    else:
        to_run = [args.experiment]

    print(f"\n  Running experiments: {to_run}")
    print(f"  Model: {MODEL_ID}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Runs per experiment: {args.runs}")

    results: list[ExperimentResult] = []
    for name in to_run:
        fn = EXPERIMENTS[name]
        result = await fn(client, args.max_tokens, args.runs)
        results.append(result)

    print_summary(results)
    save_results(results, args.output)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM prefix caching across realistic scenarios"
    )
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        default="all",
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens to generate per request (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Repeat each experiment N times (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="prefix_caching_results.json",
        help="Output JSON file (default: prefix_caching_results.json)",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
