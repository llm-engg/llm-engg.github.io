"""
benchmark_transformers_vs_vllm.py
=================================
Compare HuggingFace Transformers (offline) vs vLLM (server) on 100 prompts.

Model: cyankiwi/Qwen3.5-9B-AWQ-4bit (AWQ 4-bit quantized)

Prerequisites:
    pip install transformers autoawq torch openai accelerate

Usage:
    # 1. Start the vLLM server first (see run_vllm_server.sh)
    # 2. Run this script
    python benchmark_transformers_vs_vllm.py

    # Run only one backend:
    python benchmark_transformers_vs_vllm.py --backend transformers
    python benchmark_transformers_vs_vllm.py --backend vllm

    # Customize:
    python benchmark_transformers_vs_vllm.py --max-tokens 256 --concurrency 16
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "cyankiwi/OmniCoder-9B-AWQ-4bit"
VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MAX_TOKENS = 128
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CONCURRENCY = 32  # concurrent requests to vLLM


# ---------------------------------------------------------------------------
# 100 diverse prompts across categories
# ---------------------------------------------------------------------------

PROMPTS = [
    # --- Reasoning & Logic (1-15) ---
    "Explain the difference between inductive and deductive reasoning with examples.",
    "A bat and a ball together cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost? Show your reasoning step by step.",
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain why or why not.",
    "What is the Monty Hall problem and why is switching doors the better strategy?",
    "Explain the concept of proof by contradiction with a simple mathematical example.",
    "Three friends split a hotel room costing $30. They each pay $10. The manager realizes the room is only $25 and gives $5 to the bellboy. The bellboy keeps $2 and returns $1 to each friend. Now each friend paid $9, totaling $27, plus the $2 the bellboy kept is $29. Where is the missing dollar?",
    "Explain why correlation does not imply causation. Give a real-world example.",
    "What is Bayes' theorem and how does it update our beliefs with new evidence?",
    "A farmer has 17 sheep. All but 9 die. How many are left? Explain why people often get this wrong.",
    "Explain the halting problem in simple terms. Why is it important in computer science?",
    "What is the difference between NP-hard and NP-complete problems?",
    "Explain the pigeonhole principle and give three applications.",
    "Why is the sum of all natural numbers sometimes said to equal -1/12? Is this mathematically valid?",
    "Explain Gödel's incompleteness theorems in plain language.",
    "What is the difference between soundness and completeness in logic?",

    # --- Science & Technology (16-30) ---
    "Explain how CRISPR-Cas9 gene editing works at a high level.",
    "What is quantum entanglement and why did Einstein call it 'spooky action at a distance'?",
    "How does mRNA vaccine technology work? Explain the mechanism step by step.",
    "What is the difference between nuclear fission and nuclear fusion?",
    "Explain the greenhouse effect and how it relates to climate change.",
    "How does a transformer neural network architecture work? Explain attention mechanisms.",
    "What is the difference between supervised, unsupervised, and reinforcement learning?",
    "Explain how blockchain consensus mechanisms work. Compare Proof of Work and Proof of Stake.",
    "How does GPS determine your location? Explain the role of satellite triangulation.",
    "What is the Standard Model of particle physics? List the fundamental particles.",
    "Explain the concept of entropy in both thermodynamics and information theory.",
    "How do batteries store and release energy at the chemical level?",
    "What is the difference between RAM and ROM? How does each type of memory work?",
    "Explain the CAP theorem in distributed systems with practical examples.",
    "How does the immune system distinguish between self and non-self?",

    # --- Coding & Software Engineering (31-50) ---
    "Write a Python function to find the longest common subsequence of two strings.",
    "Explain the difference between a stack and a queue. When would you use each?",
    "Write a Python implementation of binary search that handles edge cases.",
    "What is the time complexity of quicksort in the best, average, and worst cases? Why?",
    "Explain the SOLID principles in object-oriented programming with examples.",
    "Write a Python function that checks if a binary tree is balanced.",
    "What is the difference between a process and a thread? When would you use each?",
    "Explain how garbage collection works in Python. What is reference counting?",
    "Write a Python decorator that retries a function up to 3 times on exception.",
    "What is the difference between optimistic and pessimistic locking in databases?",
    "Explain the publish-subscribe pattern. Give an example use case.",
    "Write a Python generator that yields the Fibonacci sequence indefinitely.",
    "What are the tradeoffs between SQL and NoSQL databases?",
    "Explain how consistent hashing works and why it's used in distributed systems.",
    "Write a Python function to serialize and deserialize a binary tree.",
    "What is the difference between TCP and UDP? When would you choose each?",
    "Explain the concept of eventual consistency with a practical example.",
    "Write a Python context manager that measures execution time of a code block.",
    "What is the difference between horizontal and vertical scaling?",
    "Explain how a B-tree index works in databases and why it's efficient for range queries.",

    # --- Creative & Open-ended (51-65) ---
    "Write a short story about an AI that discovers it can dream.",
    "Compose a haiku about the feeling of debugging code at 3 AM.",
    "Describe a day in the life of a mass in a black hole's accretion disk.",
    "Write a dialogue between Isaac Newton and Albert Einstein about gravity.",
    "Imagine you are a medieval scholar who just discovered a modern smartphone. Describe your reaction.",
    "Write a limerick about machine learning overfitting.",
    "Describe the taste of the color blue to someone who has never experienced synesthesia.",
    "Write a product review for a time machine from the perspective of a disappointed customer.",
    "Compose a short poem about the beauty of mathematical proofs.",
    "Write a letter from the last human on Earth to the first AI colony on Mars.",
    "Describe what the internet would look like if it were a physical place.",
    "Write a fairy tale where the villain is entropy.",
    "Imagine a world where gravity works in reverse every Tuesday. Describe a typical week.",
    "Write a motivational speech given by a rubber duck to a frustrated programmer.",
    "Describe the experience of being a single photon traveling from the Sun to Earth.",

    # --- Domain Knowledge (66-80) ---
    "Explain the difference between Type 1 and Type 2 diabetes.",
    "What are the main causes of the 2008 financial crisis?",
    "Explain how compilers differ from interpreters with specific examples.",
    "What is the difference between common law and civil law legal systems?",
    "Explain the Krebs cycle and its role in cellular respiration.",
    "What are the key differences between Keynesian and Austrian economics?",
    "Explain how public key cryptography works. Why can't you derive the private key from the public key?",
    "What is the trolley problem and what are the main ethical frameworks for analyzing it?",
    "Explain the difference between monetary policy and fiscal policy.",
    "How does the human visual cortex process information from the retina?",
    "What is the difference between civil and criminal law proceedings?",
    "Explain the mechanism of action of SSRIs in treating depression.",
    "What are the Geneva Conventions and what do they protect?",
    "Explain the water cycle and the role of evapotranspiration.",
    "What is the difference between a patent, trademark, and copyright?",

    # --- Summarization & Analysis (81-90) ---
    "Summarize the key ideas of 'Attention Is All You Need' paper in 5 bullet points.",
    "What are the main arguments for and against universal basic income?",
    "Compare and contrast the philosophies of Stoicism and Epicureanism.",
    "What are the pros and cons of microservices vs monolithic architectures?",
    "Summarize the major events of World War I in chronological order.",
    "Compare Python, Rust, and Go for building backend services. What are the tradeoffs?",
    "What are the main differences between agile and waterfall project management?",
    "Summarize the key findings from the latest IPCC climate report.",
    "Compare the educational philosophies of Montessori and traditional schooling.",
    "What are the main arguments for and against nuclear energy?",

    # --- Instruction Following & Formatting (91-100) ---
    "List exactly 5 tips for writing clean code. Number each tip.",
    "Explain the OSI model. Format your response as a markdown table with columns: Layer Number, Name, Function, Example Protocol.",
    "Write a JSON object representing a book with fields: title, author, year, genres (array), and rating.",
    "Give me a recipe for chocolate chip cookies. Use bullet points for ingredients and numbered steps for instructions.",
    "Create a comparison table of 4 popular programming languages: Python, JavaScript, Rust, and Go. Compare them on: typing, speed, memory safety, and main use case.",
    "Write exactly 3 sentences about the moon. No more, no less.",
    "Explain the water cycle using only words that a 5-year-old would understand.",
    "List the planets in our solar system from smallest to largest. Include their approximate diameter in km.",
    "Write a function docstring for a Python function called 'merge_sorted_lists' that takes two sorted lists and returns a single sorted list.",
    "Explain recursion in exactly 50 words.",
]

assert len(PROMPTS) == 100, f"Expected 100 prompts, got {len(PROMPTS)}"

PROMPTS = PROMPTS[:20]  # ensure we only have 20 prompts
# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class SingleResult:
    prompt_idx: int
    prompt: str
    output: str
    input_tokens: int
    output_tokens: int
    latency_s: float        # wall-clock time for this request
    ttft_s: float | None    # time to first token (vLLM streaming only)


@dataclass
class BenchmarkResult:
    backend: str
    total_time_s: float
    results: list[SingleResult]

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.results)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.results)

    @property
    def avg_latency_s(self) -> float:
        return sum(r.latency_s for r in self.results) / len(self.results)

    @property
    def median_latency_s(self) -> float:
        lats = sorted(r.latency_s for r in self.results)
        n = len(lats)
        if n % 2 == 1:
            return lats[n // 2]
        return (lats[n // 2 - 1] + lats[n // 2]) / 2

    @property
    def p99_latency_s(self) -> float:
        lats = sorted(r.latency_s for r in self.results)
        idx = int(len(lats) * 0.99)
        return lats[min(idx, len(lats) - 1)]

    @property
    def throughput_tok_per_s(self) -> float:
        return self.total_output_tokens / self.total_time_s if self.total_time_s > 0 else 0

    @property
    def avg_ttft_s(self) -> float | None:
        ttfts = [r.ttft_s for r in self.results if r.ttft_s is not None]
        return sum(ttfts) / len(ttfts) if ttfts else None


# ---------------------------------------------------------------------------
# Backend: HuggingFace Transformers (sequential, offline)
# ---------------------------------------------------------------------------

def _patch_shard_index_metadata():
    """Patch transformers to handle shard index files missing the 'metadata' key.
    Some quantized checkpoints (FP8, AWQ) omit this optional field."""
    try:
        import transformers.utils.hub as hub_utils
        _orig_get_checkpoint = hub_utils.get_checkpoint_shard_files

        def _patched_get_checkpoint(*args, **kwargs):
            try:
                return _orig_get_checkpoint(*args, **kwargs)
            except KeyError as e:
                if "metadata" in str(e):
                    # Re-import and monkey-patch the index loader
                    import json
                    from pathlib import Path

                    # The index file is the second positional arg or 'index_filename'
                    # Fall back: reload the index and inject an empty metadata key
                    index_file = args[1] if len(args) > 1 else kwargs.get("index_filename")
                    if index_file and Path(index_file).exists():
                        with open(index_file) as f:
                            index = json.load(f)
                        if "metadata" not in index:
                            index["metadata"] = {"total_size": 0}
                            with open(index_file, "w") as f:
                                json.dump(index, f)
                        return _orig_get_checkpoint(*args, **kwargs)
                raise

        hub_utils.get_checkpoint_shard_files = _patched_get_checkpoint
    except Exception:
        pass  # if patch fails, let it fall through to the original error


def run_transformers(prompts: list[str], max_tokens: int) -> BenchmarkResult:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f" Transformers Backend — Loading {MODEL_ID}")
    print(f"{'='*60}")

    _patch_shard_index_metadata()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    total_start = time.perf_counter()

    for i, prompt in enumerate(prompts):
        print(f"\r  [{i+1:3d}/{len(prompts)}] Generating...", end="", flush=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(model.device)
        input_len = input_ids.shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=DEFAULT_TEMPERATURE,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        t1 = time.perf_counter()

        new_tokens = output_ids[0][input_len:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        results.append(SingleResult(
            prompt_idx=i,
            prompt=prompt,
            output=output_text,
            input_tokens=input_len,
            output_tokens=len(new_tokens),
            latency_s=t1 - t0,
            ttft_s=None,  # not measurable in batch generate
        ))

    total_end = time.perf_counter()
    print(f"\r  Completed {len(prompts)} prompts in {total_end - total_start:.1f}s")

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return BenchmarkResult(
        backend="transformers",
        total_time_s=total_end - total_start,
        results=results,
    )


# ---------------------------------------------------------------------------
# Backend: vLLM (via OpenAI-compatible API, async concurrent requests)
# ---------------------------------------------------------------------------

async def _vllm_single_request(
    client,
    idx: int,
    prompt: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> SingleResult:
    """Send a single streaming request to vLLM and measure TTFT + total latency."""
    async with semaphore:
        t0 = time.perf_counter()
        ttft = None
        chunks = []

        stream = await client.completions.create(
            model=MODEL_ID,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=DEFAULT_TEMPERATURE,
            top_p=0.95,
            stream=True,
        )

        async for chunk in stream:
            if ttft is None:
                ttft = time.perf_counter() - t0
            if chunk.choices and chunk.choices[0].text:
                chunks.append(chunk.choices[0].text)

        t1 = time.perf_counter()
        output_text = "".join(chunks)

        # Estimate token counts (exact counts from vLLM usage field aren't
        # available per-chunk in streaming mode, so we approximate)
        # For input tokens, we tokenize locally
        input_tokens = len(prompt.split()) * 1.3  # rough estimate
        output_tokens = len(output_text.split()) * 1.3

        return SingleResult(
            prompt_idx=idx,
            prompt=prompt,
            output=output_text,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            latency_s=t1 - t0,
            ttft_s=ttft,
        )


async def _run_vllm_async(
    prompts: list[str],
    max_tokens: int,
    concurrency: int,
) -> BenchmarkResult:
    from openai import AsyncOpenAI

    print(f"\n{'='*60}")
    print(f" vLLM Backend — {VLLM_BASE_URL}")
    print(f" Concurrency: {concurrency}")
    print(f"{'='*60}")

    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")

    # Verify server is reachable
    try:
        models = await client.models.list()
        available = [m.id for m in models.data]
        print(f"  Available models: {available}")
    except Exception as e:
        print(f"\n  ERROR: Cannot reach vLLM server at {VLLM_BASE_URL}")
        print(f"  {e}")
        print(f"  Start the server first: bash run_vllm_server.sh")
        sys.exit(1)

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0

    async def tracked_request(idx, prompt):
        nonlocal completed
        result = await _vllm_single_request(client, idx, prompt, max_tokens, semaphore)
        completed += 1
        print(f"\r  [{completed:3d}/{len(prompts)}] Completed", end="", flush=True)
        return result

    total_start = time.perf_counter()
    tasks = [tracked_request(i, p) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    total_end = time.perf_counter()

    print(f"\r  Completed {len(prompts)} prompts in {total_end - total_start:.1f}s")

    # Get more accurate token counts via a non-streaming request for one sample
    # (optional: uncomment to calibrate)
    # resp = await client.completions.create(model=MODEL_ID, prompt="test", max_tokens=1)

    return BenchmarkResult(
        backend="vllm",
        total_time_s=total_end - total_start,
        results=sorted(results, key=lambda r: r.prompt_idx),
    )


def run_vllm(prompts: list[str], max_tokens: int, concurrency: int) -> BenchmarkResult:
    return asyncio.run(_run_vllm_async(prompts, max_tokens, concurrency))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(result: BenchmarkResult):
    print(f"\n{'─'*60}")
    print(f"  {result.backend.upper()} Results")
    print(f"{'─'*60}")
    print(f"  Total wall time      : {result.total_time_s:>10.2f} s")
    print(f"  Total output tokens  : {result.total_output_tokens:>10d}")
    print(f"  Throughput           : {result.throughput_tok_per_s:>10.1f} tok/s")
    print(f"  Avg latency/request  : {result.avg_latency_s:>10.3f} s")
    print(f"  Median latency       : {result.median_latency_s:>10.3f} s")
    print(f"  P99 latency          : {result.p99_latency_s:>10.3f} s")
    if result.avg_ttft_s is not None:
        print(f"  Avg TTFT             : {result.avg_ttft_s:>10.3f} s")
    print()


def print_comparison(tf_result: BenchmarkResult, vllm_result: BenchmarkResult):
    print(f"\n{'='*60}")
    print(f"  COMPARISON: Transformers vs vLLM")
    print(f"{'='*60}")

    headers = ["Metric", "Transformers", "vLLM", "Speedup"]
    rows = [
        (
            "Total time (s)",
            f"{tf_result.total_time_s:.1f}",
            f"{vllm_result.total_time_s:.1f}",
            f"{tf_result.total_time_s / vllm_result.total_time_s:.1f}x",
        ),
        (
            "Throughput (tok/s)",
            f"{tf_result.throughput_tok_per_s:.1f}",
            f"{vllm_result.throughput_tok_per_s:.1f}",
            f"{vllm_result.throughput_tok_per_s / tf_result.throughput_tok_per_s:.1f}x",
        ),
        (
            "Avg latency (s)",
            f"{tf_result.avg_latency_s:.3f}",
            f"{vllm_result.avg_latency_s:.3f}",
            f"{tf_result.avg_latency_s / vllm_result.avg_latency_s:.1f}x",
        ),
        (
            "Median latency (s)",
            f"{tf_result.median_latency_s:.3f}",
            f"{vllm_result.median_latency_s:.3f}",
            f"{tf_result.median_latency_s / vllm_result.median_latency_s:.1f}x",
        ),
        (
            "P99 latency (s)",
            f"{tf_result.p99_latency_s:.3f}",
            f"{vllm_result.p99_latency_s:.3f}",
            f"{tf_result.p99_latency_s / vllm_result.p99_latency_s:.1f}x",
        ),
    ]

    # Print as table
    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * w for w in col_widths)
    print(f"  {header_line}")
    print(f"  {sep_line}")
    for row in rows:
        print(f"  {' | '.join(row[i].ljust(col_widths[i]) for i in range(len(row)))}")
    print()


def save_results(results: list[BenchmarkResult], output_path: str):
    data = {}
    for r in results:
        data[r.backend] = {
            "total_time_s": r.total_time_s,
            "total_output_tokens": r.total_output_tokens,
            "throughput_tok_per_s": r.throughput_tok_per_s,
            "avg_latency_s": r.avg_latency_s,
            "median_latency_s": r.median_latency_s,
            "p99_latency_s": r.p99_latency_s,
            "avg_ttft_s": r.avg_ttft_s,
            "num_prompts": len(r.results),
            "samples": [
                {
                    "prompt_idx": s.prompt_idx,
                    "prompt": s.prompt[:100],
                    "output_preview": s.output[:200],
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    "latency_s": s.latency_s,
                    "ttft_s": s.ttft_s,
                }
                for s in r.results
            ],
        }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Transformers vs vLLM on 100 prompts"
    )
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm", "both"],
        default="both",
        help="Which backend(s) to benchmark (default: both)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens to generate per prompt (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Concurrent requests for vLLM (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=20,
        help="Number of prompts to use (default: 100, max: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file for results (default: benchmark_results.json)",
    )
    args = parser.parse_args()

    prompts = PROMPTS[: min(args.num_prompts, len(PROMPTS))]
    print(f"\nBenchmarking {len(prompts)} prompts, max_tokens={args.max_tokens}")
    print(f"Model: {MODEL_ID}")

    all_results: list[BenchmarkResult] = []

    if args.backend in ("transformers", "both"):
        tf_result = run_transformers(prompts, args.max_tokens)
        print_report(tf_result)
        all_results.append(tf_result)

    if args.backend in ("vllm", "both"):
        vllm_result = run_vllm(prompts, args.max_tokens, args.concurrency)
        print_report(vllm_result)
        all_results.append(vllm_result)

    if args.backend == "both" and len(all_results) == 2:
        print_comparison(all_results[0], all_results[1])

    save_results(all_results, args.output)


if __name__ == "__main__":
    main()
