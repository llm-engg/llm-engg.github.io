from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import time
import sys


target_name = "Qwen/Qwen2.5-3B-Instruct"
assistant_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(target_name)

# Load target model
model = AutoModelForCausalLM.from_pretrained(
    target_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load assistant model
assistant = AutoModelForCausalLM.from_pretrained(
    assistant_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Baseline
print("Baseline (no speculative decoding)")
start = time.time()
_ = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False,
    streamer=streamer,
)
baseline_time = time.time() - start
print(f"\nBaseline: {baseline_time:.2f}s")

input("\nPress Enter to run speculative decoding...\n")

# Speculative decoding
print("Speculative Decoding")
start = time.time()
_ = model.generate(
    **inputs,
    assistant_model=assistant,
    max_new_tokens=100,
    do_sample=False,
    streamer=streamer,
)
spec_time = time.time() - start

print(f"\nSpeculative: {spec_time:.2f}s")
print(f"Speedup: {baseline_time/spec_time:.2f}x")