# Modern Architectures

# Outline and goals

*   Quick recap of the ‘standard’ transformer (what you implement)
*   What do most of the large LMs have in common?
*   What are common variations to the architecture / training process?

**Today’s theme:** the best way to learn is hands-on experience
the second best way is to try to learn from others’ experience

---

# Starting point: the ‘original’ transformer

**Review: choices in the standard transformer**

**Position embedding:** sines and cosines
$$ PE_{(pos,2i)} = sin(pos/10000^{2i/d_{\text{model}}}) $$
$$ PE_{(pos,2i+1)} = cos(pos/10000^{2i/d_{\text{model}}}) $$

**FFN:** ReLU
$$ \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 $$

**Norm type:** post-norm, LayerNorm

*[Diagram showing the standard transformer architecture with Attention, Feed Forward, Add & Norm blocks]*

---

# What we implemented – GPT- 2 (Decoder-only transformer)

**Differences from original transformer:**
*   **LayerNorm** is in front of the block (pre-norm)
*   **Learned absolute position embeddings** (not sinusoidal)
*   FF layers use **GeLU activation**, not ReLU
*   **Bias terms included** in linear layers and LayerNorm

*[Diagram showing GPT-2 architecture with Pre-Norm, Learned Position Embeddings, and GeLU]*

**What do current models use?**

---

# How should we think about architectures?

Lots of architecture. We'll cover few of them in the next class. 

---

# Let’s look at the data (on dense architectures)

**Learn from the many other models (and papers) out there**

*[Large table screenshot listing various models (GPT, T5, LLaMA, Mistral, etc.) and their specifications]*

**We will talk through many major architecture and hyperparameter variants.**

*   What do all these models have in common?
*   What parts vary?
*   What can we learn from this?

---

# What are we going to cover?

**Common architecture variations**
*   Activations, FFN
*   Attention variants
*   Position embeddings

**Hyperparameters that (do or don’t) matter**
*   What is `ff_dim`? Do `multi_head` dims always sum to `model_dim`?
*   How many vocab elements?

---

# Architecture variations..

**Let’s think about the core architecture piece**

*[Table highlighting Norm, Position embedding, and Activations columns]*

---

# Pre-vs-post norm


*[Diagram comparing Post-LN Transformer vs Pre-LN Transformer]*

**Post-LN Transformer**

$x_{l,i}^{post,1} = \text{MultiHeadAtt}(x_{l,i}^{post}, [x_{l,1}^{post}, \dots, x_{l,n}^{post}])$

$x_{l,i}^{post,2} = x_{l,i}^{post} + x_{l,i}^{post,1}$

$x_{l,i}^{post,3} = \text{LayerNorm}(x_{l,i}^{post,2})$

$x_{l,i}^{post,4} = \text{ReLU}(x_{l,i}^{post,3}W^{1,l} + b^{1,l})W^{2,l} + b^{2,l}$

$x_{l,i}^{post,5} = x_{l,i}^{post,3} + x_{l,i}^{post,4}$

$x_{l+1,i}^{post} = \text{LayerNorm}(x_{l,i}^{post,5})$

**Pre-LN Transformer**

$x_{l,i}^{pre,1} = \text{LayerNorm}(x_{l,i}^{pre})$

$x_{l,i}^{pre,2} = \text{MultiHeadAtt}(x_{l,i}^{pre,1}, [x_{l,1}^{pre,1}, \dots, x_{l,n}^{pre,1}])$

$x_{l,i}^{pre,3} = x_{l,i}^{pre} + x_{l,i}^{pre,2}$

$x_{l,i}^{pre,4} = \text{LayerNorm}(x_{l,i}^{pre,3})$

$x_{l,i}^{pre,5} = \text{ReLU}(x_{l,i}^{pre,4}W^{1,l} + b^{1,l})W^{2,l} + b^{2,l}$

$x_{l+1,i}^{pre} = x_{l,i}^{pre,3} + x_{l,i}^{pre,5}$

Final LayerNorm: 

$x_{Final,i}^{pre} \leftarrow \text{LayerNorm}(x_{L+1,i}^{pre})$

Set up LayerNorm so that it doesn’t affect the main residual signal path (on the left)

**Almost all modern LMs use pre-norm**

---

# New things – ‘double’ norm.

**If putting LayerNorms in residual streams is bad.. Why not post-norm outside the stream?**

*[Diagram showing LayerNorm added after the addition step, but outside the main residual path]*

**Recent models:** Grok, Gemma 2. Olmo 2 *only* does non-residual post norm

---

# LayerNorm vs RMSNorm

**Original transformer: LayerNorm** – normalizes the mean and variance across $d_{\text{model}}$
$$ y = \frac{x - E[x]}{\sqrt{\text{Var}[x] + \epsilon}} * \gamma + \beta $$
**Notable models:**
GPT3/2/1, OPT, GPT-J, BLOOM

**Many modern LMs: RMSNorm** – does not subtract mean or add a bias term
$$ y = \frac{x}{\sqrt{||x||_2^2 + \epsilon}} * \gamma $$
**Notable models:**
LLaMA-family, PaLM, Chinchilla, T5

---

# Why RMSNorm?

**Modern explanation – it’s faster (and just as good).**
*   **Fewer operations** (no mean calculation)
*   **Fewer parameters** (no bias term to store)

$$ y = \frac{x - E[x]}{\sqrt{\text{Var}[x] + \epsilon}} * \gamma + \beta $$

**Does this explanation make sense?**

| Operator class | % flop |
| :--- | :--- |
| $\Delta$ Tensor contraction | 99.80 |
| $\square$ Stat. normalization | 0.17 |
| $\bigcirc$ Element-wise | 0.03 |

Matrix multiplies are the *vast* majority of FLOPs (and memory)
[Ivanov et al 2023]

---

# Why RMSNorm (2)

**Important lesson:** FLOPS are not runtime! (we will discuss this in far more detail later)

[Ivanov et al 2023]

| Operator class | % flop | % Runtime |
| :--- | :--- | :--- |
| $\Delta$ Tensor contraction | 99.80 | 61.0 |
| $\square$ Stat. normalization | 0.17 | 25.5 |
| $\bigcirc$ Element-wise | 0.03 | 13.5 |

*[Diagram showing arithmetic intensity]*
Left top ("43G") is FLOPS
Right top ("153") is the FLOP-to-memory ratio

**RMSNorm can still matter due to the importance of *data movement***

---

# More generally: dropping bias terms

**Most modern transformers don’t have bias terms.**

Original Transformer:
$$ \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 $$

Most implementations (if they’re not gated):
$$ \text{FFN}(x) = \sigma(xW_1)W_2 $$

**Reasons:** memory (similar to RMSnorm) and optimization stability

---

# LayerNorm: recap

*   **Basically everyone does pre-norm.**
    *   Intuition – keep the good parts of residual connections
    *   Observations – nicer gradient propagation, fewer spike
    *   Some people add a second norm outside the residual stream (NOT post-norm)

*   **Most people do RMSnorm**
    *   In practice, works as well as LayerNorm
    *   But, has fewer parameters to move around, which saves on wallclock time
    *   People more generally drop bias terms since the compute/param tradeoffs are not great.

---

# Activations

**A whole zoo of activations ..**

ReLU, GeLU, Swish, ELU, GLU, GeGLU, ReGLU, SeLU, SwiGLU, LiGLU

**What are these things? What do people use? Does it matter?**

---

# A few of the common activations

**ReLU**
$$ FF(x) = \max(0, xW_1) W_2 $$
*[Graph of ReLU]*
**Notable models:**
Original transformer, T5, Gopher, Chinchilla, OPT

**GeLU**
$$ FF(x) = \text{GELU}(xW_1)W_2 $$
$$ GELU(x) := x\Phi(x) $$
*[Graph of GeLU]*
**Notable models:**
GPT1/2/3, GPTJ, GPT-Neox, BLOOM

**SwiGLU / GeGLU (next slide..)**
**Notable models:**
Llama, PaLM, T5 v1.1, *most models post 2023*

---

# Gated activations (*GLU)

**GLUs modify the ‘first part’ of a FF layer**
$$ FF(x) = \max(0, xW_1) W_2 $$

**Instead of a linear + ReLU, augment the above with an (entrywise) linear term**
$$ \max(0, xW_1) \rightarrow \max(0, xW_1) \otimes (xV) $$

**This gives the gated variant (ReGLU) – note that we have an extra parameter (V)**
$$ \text{FF}_{\text{ReGLU}}(x) = (\max(0, xW_1) \otimes xV) W_2 $$

---

# Gated variants of standard FF layers

**GeGLU**
$$ \text{FFN}_{\text{GEGLU}}(x, W, V, W_2) = (\text{GELU}(xW) \otimes xV)W_2 $$
**Notable models:**
T5 v1.1, mT5, LaMDA, Phi3, Gemma 2, Gemma 3

**SwiGLU (swish is $x * \text{sigmoid}(x)$)**
$$ \text{FFN}_{\text{SwiGLU}}(x, W, V, W_2) = (\text{Swish}_1(xW) \otimes xV)W_2 $$
**Notable models:**
LLaMa 1/2/3, PaLM, Mistral, OlMo, *most models post 2023*

Note: Gated models use smaller dimensions for the $d_{ff}$ by 2/3

---

# Serial vs Parallel layers

**Normal transformer blocks are serial – they compute attention, then the MLP**

*[Diagram of Serial Transformer Block]*
Add -> Dropout -> Feed-Forward -> Norm -> Add -> Dropout -> Attention -> Norm

**Could we parallelize the transformer block?**

---

# Parallel layers

**A few models (GPTJ, PaLM, GPT-NeoX) do parallel layers. Originally in GPT-J**

**Parallel Layers** – We use a "parallel" formulation in each Transformer block (Wang & Komatsuzaki, 2021), rather than the standard "serialized" formulation. Specifically, the standard formulation can be written as:
$$ y = x + \text{MLP}(\text{LayerNorm}(x + \text{Attention}(\text{LayerNorm}(x))) $$
Whereas the parallel formulation can be written as:
$$ y = x + \text{MLP}(\text{LayerNorm}(x)) + \text{Attention}(\text{LayerNorm}(x)) $$
The parallel formulation results in roughly 15% faster training speed at large scales, since the MLP and Attention input matrix multiplications can be fused. Ablation experiments showed a small quality degradation at 8B scale but no quality degradation at 62B scale, so we extrapolated that the effect of parallel layers should be quality neutral at the 540B scale.

**If implemented right, LayerNorm can be shared, and matrix multiplies can be fused**

**Recent Models:** Cohere Command A, Falcon 2 11B, Command R+

---

# Summary: architectures

**Pre-vs-post norm:**
*   Everyone does pre-norm (except OPT350M), likely with good reason.

**Layer vs RMSnorm:**
*   RMSnorm has clear compute wins, sometimes even performance

**Gating:**
*   GLUs seem generally better, though differences are small

**Serial vs parallel layers:**
*   No extremely serious ablations, but has a compute win.

*[Table summary of architectures visible on the right]*

---

# Many variations in position embeddings



**Sine embeddings:** add sines and cosines that enable localization
$$ Embed(x, i) = v_x + PE_{pos} $$
$$ PE_{(pos,2i)} = sin(pos/10000^{2i/d_{\text{model}}}) $$
$$ PE_{(pos,2i+1)} = cos(pos/10000^{2i/d_{\text{model}}}) $$
**Notable models:**
Original transformer

**Absolute embeddings:** add a position vector to the embedding
$$ Embed(x, i) = v_x + u_i $$
**Notable models:**
GPT1/2/3, OPT

**Relative embeddings:** add a vector to the attention computation
$$ e_{ij} = \frac{x_i W^Q (x_j W^K + a_{ij}^K)^T}{\sqrt{d_z}} $$
**Notable models:**
T5, Gopher, Chinchilla

**Rope embeddings** (next slides..)
**Notable models:**
GPTJ, PaLM, LLaMA
*Most 2024+ models*

---

# RoPE: rotary position embeddings

**High level thought process: a relative position embedding should be some $f(x, i)$ s.t.**
$$ \langle f(x, i), f(y, j) \rangle = g(x, y, i - j) $$
That is, the attention function *only* gets to depend on the relative position (i-j). How do existing embeddings not fulfill this goal?

*   **Sine:** Has various cross-terms that are not relative
    $$ \langle Embed(x, i), Embed(y, i) \rangle = \langle v_x, v_y \rangle + \langle PE_i, v_y \rangle ... $$
*   **Absolute:** obviously not relative
*   **Relative embeddings:**
    $$ e_{ij} = \frac{x_i W^Q (x_j W^K + a_{ij}^K)^T}{\sqrt{d_z}} $$
    is not an inner product

---

# RoPE: rotary position embeddings

**How can we solve this problem?**
*   We want our embeddings to be invariant to absolute position
*   We know that inner products are invariant to arbitrary rotation.

*[Diagram illustrating vectors rotating]*
Position independent embedding -> Rotate "we" by '0 positions', "know" by '1 position' -> Rotate "we" by '2 positions', "know" by '3 positions'. The relative angle between vectors remains constant.

---

# RoPE: rotary position embeddings

**There are many rotations, which one do you pick?**

*[Diagram showing rotation in 2D pairs]*
Just pair up the coordinates and rotate them in 2d (motivation: complex numbers)

[Su et al 2021]

---

# The actual RoPE math

**Multiply with sines and cosines**

$$ f_{\{q,k\}}(x_m, m) = \boldsymbol{R}_{\Theta,m}^d \boldsymbol{W}_{\{q,k\}} x_m \quad (14) $$

$$ \boldsymbol{R}_{\Theta,m}^d = \begin{pmatrix} \cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\ \sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\ 0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0 \\ 0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2} \\ 0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2} \end{pmatrix} \quad (15) $$

**Difference with sine embeddings – not additive, no cross terms**

---

# Implementation and code for RoPE

```python
query_states = self.q_proj(hidden_states)
key_states = self.k_proj(hidden_states)
value_states = self.v_proj(hidden_states)

# Flash attention requires the input to have the shape
# batch_size x seq_length x head_dim x hidden_dim
# therefore we just need to keep the original shape
query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

cos, sin = self.rotary_emb(value_states, position_ids)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

...
```
Same stuff as the usual multi-head self attention below

**Note: embedding at *each attention operation* to enforce position invariance**

---

# Hyperparameters

Transformer hyperparameter questions you might have had in 224n..

*   How much bigger should the feedforward size be compared to hidden size?
*   How many heads, and should num_heads always divide hidden size?
*   What should my vocab size be?

**And other model setting questions**
*   Do people even regularize these huge LMs?
*   How do people scale these models - very deep or very wide?

---


**Feedforward – model dimension ratio.**

$$ \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 $$

There are two dimensions that are relevant – the feedforward dim ($d_{ff}$) and model dim ($d_{model}$). What should their relationship be?

$$ \boldsymbol{d_{ff} = 4 d_{model}} $$

This is *almost always* true. There’s just a few exceptions.

---

# Surprising (?) consensus hyperparameter 2

**Head-dim*num-heads to model-dim ratio. As a reminder, slide from 224n.**

> **Multi-head self-attention is computationally efficient**
> *   Even though we compute $h$ many attention heads, it's not really more costly.
>     *   We compute $XQ \in \mathbb{R}^{n \times d}$, and then reshape to $\mathbb{R}^{n \times h \times d/h}$. (Likewise for $XK, XV$.)

> *   The total cost is still $O(n^2 d)$, same as single-head attention with dimension $d$.
---


# What are typical vocabulary sizes?

**Monolingual models – 30-50k vocab**

| Model | Token count |
| :--- | :--- |
| Original transformer | 37000 |
| GPT | 40257 |
| GPT2/3 | 50257 |
| T5/T5v1.1 | 32128 |
| LLaMA | 32000 |

**Multilingual / production systems 100-250k**

| Model | Token count |
| :--- | :--- |
| mT5 | 250000 |
| PaLM | 256000 |
| GPT4 | 100276 |
| Command A | 255000 |
| DeepSeek | 100000 |
| Qwen 15B | 152064 |
| Yi | 64000 |

**Monolingual vocabs don’t need to be huge, but multilingual ones do**

---

# Dropout and other regularization

**Do we need regularization during pretraining?**

**Arguments against:**
*   There is *a lot* of data (trillions of tokens), more than parameters.
*   SGD only does a single pass on a corpus (hard to memorize)

This is all quite reasonable.. but what do people do in practice?

---

# Summary: hyperparameters

**Feedforward**
*   Factor-of-4 rule of thumb (8/3 for GLUs) is standard (with some evidence)

**Head dim**
*   Head dim*Num head = D model is standard – but low to no validation

---

# Attention heads

**GQA / MQA :** Saving inference costs by reducing the number of heads

**Sparse or sliding window attention (GPT4/Mistral):** restricting the attention pattern to reduce compute cost

---

# GQA/MQA – Reducing attention head cost

**Let’s think about the compute involved for attention**

*[Diagram showing attention calculation $XQ K^T X^T$]*

$$ \text{softmax} \left( X Q K^T X^T \right) XV = P \cdot V = \text{output} \in \mathbb{R}^{n \times d} $$

**Total arithmetic operations ($bnd^2$), total memory accesses ($bnd + bhn^2 + d^2$)**

Arithmetic intensity is high $O \left( \left(\frac{1}{k} + \frac{1}{bn} \right)^{-1} \right)$ - we can keep our GPUs running

---

# GQA/MQA – Reducing attention head cost

**What about the *incremental* case when we generate text?**

**Key difference:** can’t parallelize the generation process – needs to be step by step

**In this case – we need to incrementaly re-compute/update attention via the ‘KV cache’**

*[Diagram showing KV Caching process]*
[Animation from https://medium.com/@joaolages/kv-caching-explained-276520203249]

---

# GQA/MQA – Reducing attention head cost

**What’s the incremental arithmetic intensity?**

**Total arithmetic operations ($bnd^2$), total memory accesses ($bn^2d + nd^2$)**

Arithmetic intensity is not good $O \left( \left(\frac{n}{d} + \frac{1}{b} \right)^{-1} \right)$ - need large batches + short seq length (n) or big model dimensions (d)

**Is there some way around this? The n/d term is difficult to reduce.**

---

# MQA – just have fewer key dimensions.

**Key idea – have multiple queries, but just one dimension for keys and values**

*[Diagram showing Multi-Query Attention with shared Keys and Values]*

**We have much fewer items to move in and out of memory (KV Cache)**

**Total memory access ($bnd + bn^2k + nd^2$), Arithmetic intensity $O \left( \left(\frac{1}{d} + \frac{n}{dh} + \frac{1}{b} \right)^{-1} \right)$**

[figure from https://blog.fireworks.ai/multi-query-attention-is-all-you-need-db072e758055]

---

# Recent extension – GQA

**Don’t go all the way to one dimension of KV – have fewer dims**

*[Diagram comparing Multi-head, Grouped-query, and Multi-query attention]*

**Simple knob to control expressiveness (key-query ratio) and inference efficiency**

---

# Does MQA hurt? Sometimes..

**Small PPL hit w/ MQA [Shazeer 2019]**

| Attention | $h$ | $d_k, d_v$ | $d_{ff}$ | dev-PPL |
| :--- | :--- | :--- | :--- | :--- |
| multi-head | 8 | 128 | 8192 | **29.9** |
| multi-query | 8 | 128 | 9088 | 30.2 |
| multi-head | 1 | 128 | 9984 | 31.2 |
| multi-head | 2 | 64 | 9984 | 31.1 |
| multi-head | 4 | 32 | 9984 | 31.0 |
| multi-head | 8 | 16 | 9984 | 30.9 |

**Low/no hit w/ GQA [Ainslie 2023]**
*[Graphs showing Performance vs Time per sample]*

---

# Sparse / sliding window attention

**Attending to the entire context can be expensive (quadratic).**

**Build sparse / structured attention that trades off expressiveness vs runtime (GPT3)**

*[Diagrams showing attention matrices: (a) Transformer (full), (b) Sparse Transformer (strided), (c) Sparse Transformer (fixed)]*

[Child et al 2019]

---

# Sliding window attention

**Another variation on this idea – sliding window attention**

*[Diagram comparing Vanilla Attention vs Sliding Window Attention]*

**Just use the main part of the strided pattern – let depth extend effective context (Mistral)**

---

# Current standard trick – interleave ‘full’ and ‘LR’ attention

**From Cohere Command A – Every $4^{\text{th}}$ layer is a full attention**

*[Diagram showing Command A Transformer Block sequence: SWA -> SWA -> SWA -> Full]*

**Long-range info via NoPE, short-range info via RoPE + SWA.**

**Other models** – LLaMA 4, Gemma does SWA+Full RoPE.

---

# Recap, conclusion, etc.

**Many aspects (arch, hparams) of transformers are in common across the big LMs**

*[Large table summarizing model parameters]*

**Major differences? Position embeddings, activations, tokenization**


--- 
---

# LLM Architecture Evolution: 2023 → 2025

**Big Picture: From scaling dense Transformers → engineering-driven efficiency**

*   MoE maturity
*   Long-context stability
*   Active parameters ≠ total parameters

---

# 2023: Scaling Era (Baseline Transformers)

**Representative models:** GPT-4, LLaMA-2

| Component | State |
| :--- | :--- |
| **Architecture** | Dense Transformers dominate<br>Sequential Attention → MLP blocks |
| **Normalization** | Pre-Norm becomes standard (Post-Norm effectively dead)<br>Mix of LayerNorm and early RMSNorm |
| **FFN / Activations** | Transition to SwiGLU underway but not universal |
| **Position Encoding** | RoPE standard, limited long-context stability |
| **Context** | 8k–32k typical, 128k is exceptional |
| **Training** | BF16/FP16<br>FLOPs scale ≈ "just make it bigger" |

**Key limitation:** Inefficient scaling: cost ∝ parameters

---

# 2024: Efficiency & Structure Take Over

**Representative models:** DeepSeek-V3, Mistral-Small-3, Grok-2.5

| Component | State |
| :--- | :--- |
| **Architecture** | MoE returns seriously (not research toys anymore)<br>Hybrid attention (GQA, sliding window) |
| **Normalization** | RMSNorm wins (LayerNorm mostly gone) |
| **FFN / Activations** | SwiGLU is default<br>Shared experts appear in MoE |
| **Position Encoding** | NTK-scaled / extended RoPE |
| **Context** | 32k–128k becomes normal |
| **Training** | Still BF16, but FLOPs efficiency matters |

**Key shift:** Active parameters ≠ total parameters. Cost/performance optimization mindset

---

# 2025: Engineering-First LLMs

**Representative models:** DeepSeek-R1, Llama-4, Qwen-3, Gemini-3

| Component | State |
| :--- | :--- |
| **Architecture** | MoE is mature (routing, shared experts, stability solved)<br>Dense models still exist, but only when justified |
| **Block Design** | Experimental parallel Attention + MLP appears<br>Kernel-fusion-friendly layouts |
| **Normalization** | RMSNorm + Pre-Norm is universal |
| **Position Encoding** | Long-context-safe RoPE variants are mandatory |
| **Context** | 128k is baseline<br>Frontier models: 200k → 1M tokens |
| **Training** | FP8 adoption begins<br>FLOPs/token is the real metric |

---
