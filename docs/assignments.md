<!-- # Assignments

This page contains all course assignments, projects, and hands-on exercises. Each assignment is designed to reinforce the concepts learned in the corresponding week and build practical skills in LLM engineering.

## Assignment Structure

Each week features different types of learning activities:

- **Lab Exercises**: Hands-on coding and implementation
- **Case Studies**: Analysis of real-world LLM applications  
- **Projects**: Progressive building of LLM systems
- **Experiments**: Research and exploration tasks

## Week-by-Week Assignments

### Foundation Phase

#### Week 1: Transformer Architecture
- **Lab**: Attention Visualization - implement and visualize attention mechanisms
- **Project**: Code walkthrough of Transformer implementation
- **Exercise**: Analyze GPT-2 architecture and training process

#### Week 2: Tokenization & MoE
- **Lab**: Tokenization playground - experiment with different tokenizers
- **Project**: Pretrain a tiny language model from scratch
- **Case Study**: Compare MoE inference vs dense inference performance

#### Week 3: Modern Architectures
- **Lab**: Modern LLM case study using Hugging Face models
- **Project**: Compare configs and forward passes of LLaMA, Mistral, and Qwen
- **Exercise**: Visualize differences in architecture (attention heads, layer norms)

### Infrastructure Phase

#### Week 4: GPU Programming
- **Lab**: Write CUDA kernels (vector add, matrix multiply) and benchmark vs PyTorch
- **Project**: Explore GPU profiling tools (nvprof, Nsight)
- **Case Study**: Large GPU cluster cost analysis for training DeepSeek/GPT-5

### Inference Optimization Phase

#### Week 5: Inference Fundamentals
- **Lab**: Use PyTorch Profiler to track memory vs compute bottlenecks
- **Project**: Explore Hugging Face Hub & inference APIs
- **Exercise**: Serve a model with vLLM

#### Week 6: Quantization
- **Lab**: Quantize models with GPTQ using AutoGPTQ or bitsandbytes
- **Project**: Compare fp16, int8, int4 inference (tokens/sec, VRAM use, perplexity)
- **Research**: Explore hallucinations induced by quantized models

#### Week 7: Multi-GPU Serving
- **Lab**: Benchmark vLLM vs HF TGI on the same model
- **Project**: Deploy a model on multi-GPU using vLLM
- **Case Study**: Deploy DeepSeek model on 100 GPUs

### Fine-tuning Phase

#### Week 8: PEFT Methods
- **Lab**: Fine-tune SMOL-LM3 on a simple instruction-following task
- **Project**: Compare LoRA vs QLoRA performance and efficiency
- **Research**: Investigate catastrophic forgetting mitigation

#### Week 9: Instruction Tuning & Alignment
- **Lab**: Explore preference datasets like Anthropic/hh-rlhf
- **Project**: Take SFT model and align it using DPO
- **Exercise**: Generate synthetic preference pairs

#### Week 10: Reasoning
- **Lab**: Show impact of CoT on reasoning accuracy (GSM8k, coding tasks)
- **Project**: Fine-tune a small model on reasoning dataset
- **Research**: Compare different reasoning strategies

### Applications Phase

#### Week 11: RAG Systems
- **Lab**: Build a small RAG pipeline with FAISS vector DB
- **Project**: Evaluate retrieval quality using RAGas metrics
- **Case Study**: Implement Graph RAG for complex documents

#### Week 12-13: AI Agents
- **Lab**: Implement a toy MCP wrapper around a model agent
- **Project**: Fine-tune a small model on tool-augmented dataset
- **Exercise**: Evaluate improvement in reasoning + tool accuracy

### Advanced Topics Phase

#### Week 14: Evaluation & Monitoring
- **Lab**: Implement LLM-as-a-Judge evaluation
- **Project**: Monitor a live RAG/agent pipeline
- **Exercise**: Set up A/B testing for model comparison

#### Week 15: Multimodal Models
- **Lab**: Run a Large Multimodal Model (LMM) locally
- **Project**: Fine-tune LLaVA-Phi-3 or Moondream on custom visual task
- **Exercise**: Test Visual Question Answering capabilities

#### Week 16: Edge Deployment
- **Lab**: Run LLM on Raspberry Pi
- **Project**: Deploy Gemma-3n on Android/iOS
- **Exercise**: Fine-tune Gemma-3n for Kannada language

#### Week 17: Security & Frontiers
- **Lab**: Demonstrate prompt injection attack and defense
- **Project**: Deploy Qwen-3 Next for production use
- **Exercise**: Run Mamba-7B on 100K context and compare to Llama-3

### Final Phase

#### Week 18: Student Presentations
- **Project**: Final project presentations
- **Peer Review**: Evaluate and discuss peer projects
- **Reflection**: Course wrap-up and future learning paths

## Submission Guidelines

### Code Submissions
- All code must be submitted via GitHub repositories
- Include comprehensive README with setup instructions
- Provide requirements.txt or environment.yml files
- Include documentation and comments

### Reports
- Technical reports should be 2-3 pages maximum
- Include performance metrics, charts, and analysis
- Compare results with baseline or existing methods
- Discuss limitations and future improvements

### Presentations
- 10-15 minute presentations for final projects
- Include live demos when possible
- Focus on practical insights and lessons learned

## Grading Criteria

- **Technical Implementation** (40%): Code quality, correctness, efficiency
- **Analysis & Insights** (30%): Understanding of concepts, thoughtful analysis
- **Documentation** (20%): Clear explanations, reproducible results
- **Innovation** (10%): Creative approaches, going beyond requirements

## Resources & Support

- **Office Hours**: Weekly Q&A sessions for assignment help
- **Discussion Forum**: Peer collaboration and problem-solving
- **Computing Resources**: Access to GPU clusters for large model experiments
- **Reference Materials**: Curated papers, blogs, and documentation

---

*Assignments will be released weekly and are due before the start of the following week's session. Late submissions will be penalized unless prior arrangements are made.* -->

**TBD**