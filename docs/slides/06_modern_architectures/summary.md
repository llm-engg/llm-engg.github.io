# Modern Transformer Architecture Variants

## Overview & Methodology

The lecture takes a **data-driven, empirical approach** to understanding transformer architecture choices. Rather than theorizing about what should work, Hashimoto surveyed **19+ dense model releases** from 2017-2025 (Original Transformer through SmolLM, Command A, Gemma 3) and compiled a spreadsheet tracking architecture decisions across all of them. The key insight: you can observe **convergent evolution** in neural architectures -- certain choices have won out decisively, while others remain genuinely open.

**Central theme:** "The best way to learn is hands-on experience; the second best way is to learn from others' experience."

---

## Part 1: Architecture Variations

### 1.1 Pre-Norm vs Post-Norm (Universal consensus)

**The one thing everyone agrees on.** This is the single strongest consensus in modern LLM architecture.

**What changed:** The original transformer (Vaswani 2017) placed LayerNorm *after* the residual addition (post-norm). Pre-norm moves it *before* the sub-layer, keeping the residual stream clean.

**Post-LN:** `x → Attention(x) → Add → LayerNorm → FFN → Add → LayerNorm`
**Pre-LN:** `x → LayerNorm → Attention → Add → LayerNorm → FFN → Add`

**Why pre-norm wins:**
- **Gradient propagation:** The residual stream provides an identity connection from top to bottom. Post-norm inserts normalization into this path, disrupting gradient flow. Xiong (2020) showed post-LN has gradient attenuation that grows with depth -- at initialization, gradients explode in later layers; only after careful warmup does it stabilize.
- **Training stability:** Salazar & Nguyen (2019) showed post-norm training exhibits frequent gradient spikes (visible as high-variance gradient norms). Pre-norm dramatically reduces these spikes.
- **Practical benefit today:** Pre-norm allows using larger learning rates and eliminates the need for careful warmup schedules. This matters enormously for large-scale training.


#### Teaching Intuition

Think of the residual stream as a **highway** running from the bottom of the network to the top. In post-norm, you're placing toll booths (LayerNorms) directly on the highway -- every car (gradient) has to stop and be processed. In pre-norm, you've moved the toll booths onto the exit ramps (the attention/FFN branches). The highway itself is completely clear, so gradients travel freely end-to-end.

A concrete way to show this: draw the computational graph and trace the gradient path. In post-norm, the gradient must pass through LayerNorm's Jacobian at every layer (multiplied L times for L layers). In pre-norm, there's always a direct additive path through the residual that bypasses everything.

#### Student Questions (from the CS336 lecture)

**Q: Why exactly is LayerNorm in the residual stream bad?**
A: The residual gives you an identity connection from almost the top to the bottom of the network. This makes gradient propagation trivially easy -- there's no vanishing/exploding gradient along this path. Putting LayerNorm in the middle disrupts this identity. The empirical evidence confirms it: post-norm models show gradient attenuation (gradients grow across layers at init) and require careful warmup to stabilize.

**Q: If pre-norm preserves the residual, don't the activations grow unboundedly as they flow through layers?**
A: Yes, they can! This is actually a real concern. The residual stream accumulates contributions from every layer, and its magnitude can grow. This is part of why a **final LayerNorm** is needed before the output projection in pre-norm architectures. It's also part of the motivation for "double norm" -- adding normalization on the branch outputs controls their magnitude before they enter the residual.

#### Advanced Questions for Students

1. **Derive it:** Write out the backward pass for a 2-layer post-norm vs pre-norm network. Show explicitly where the LayerNorm Jacobian appears in each case and how it affects gradient magnitude.

2. **The double-norm puzzle:** If pre-norm works because we keep the residual clean, then double-norm (pre + post outside residual) also keeps the residual clean. But now the branch output is normalized twice. Could this hurt expressiveness? What's the tradeoff?

3. **Thought experiment:** BERT was trained with post-norm and was hugely successful. Why did post-norm work for BERT but became problematic for GPT-scale models? (Hint: think about depth, training duration, and learning rate schedules.)

**New development -- "Double norm" (2024-2025):**
If the intuition is "keep LayerNorms out of the residual stream," why not add a *second* norm after the sub-layer but outside the residual path? Recent models do exactly this:
- **Grok, Gemma 2:** LayerNorm both before AND after attention/FFN blocks (outside residual)
- **OLMo 2:** Only the post-norm outside the residual stream (not the pre-norm)

This is argued to further improve stability for very large models.

### 1.2 LayerNorm vs RMSNorm (Strong consensus toward RMSNorm)

**LayerNorm** (original): Normalize by subtracting mean and dividing by std dev, then scale ($\gamma$) and shift ($\beta$):
$$y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} \cdot \gamma + \beta$$

**RMSNorm** (modern): Drop the mean subtraction and bias term:
$$y = \frac{x}{\sqrt{||x||_2^2 + \epsilon}} \cdot \gamma$$

**Notable models using LayerNorm:** GPT-1/2/3, OPT, GPT-J, BLOOM
**Notable models using RMSNorm:** LLaMA family, PaLM, Chinchilla, T5

**Why RMSNorm wins -- a nuanced argument:**

The naive argument is "fewer operations = faster." But normalization is only 0.17% of FLOPs (tensor contractions are 99.8%). So saving a mean calculation seems meaningless.

**The real reason: FLOPS are not runtime.** (Key insight from Ivanov et al 2023)

| Operator class | % FLOPs | % Runtime |
|---|---|---|
| Tensor contraction | 99.80 | 61.0 |
| Stat. normalization | 0.17 | **25.5** |
| Element-wise | 0.03 | 13.5 |

Normalization operations are 0.17% of FLOPs but **25.5% of runtime** because they are dominated by **data movement** (memory bandwidth), not compute. RMSNorm has fewer parameters to move in and out of memory, which translates to real wallclock savings.

**Empirical validation (Narang et al 2020):** RMSNorm achieves both:
- Faster training (3.68 vs 3.50 steps/second)
- Lower final loss (1.821 vs 1.838)
- Better downstream performance (SGLUE 75.45 vs 71.66)

#### Teaching Intuition

Show students the FLOP breakdown table and ask: "Should we bother optimizing normalization if it's only 0.17% of FLOPs?" Let them argue "no." Then reveal the runtime column -- normalization is **25% of wallclock time**. This is the single most important systems insight for architecture design: **FLOPs are a terrible proxy for runtime.** Memory bandwidth, not compute, is the bottleneck for many operations. This lesson recurs throughout inference optimization, kernel fusion, and GPU architecture.

An analogy: Imagine a factory where 99.8% of the work is done by robots (matrix multiplies) and 0.2% by humans (normalization). The robots are incredibly fast. The humans aren't slow at their task, but they spend most of their time *walking to the warehouse to fetch parts* (memory access). Optimizing the human's work saves far more time than you'd expect from the 0.2% number.

#### Student Questions

**Q: If mean-centering doesn't matter, does that tell us something about what LayerNorm actually does?**
A: Great question. It suggests that the *scale normalization* (dividing by magnitude) is the critical operation, not the centering. The model can learn to handle non-zero-mean activations just fine. What it really needs is for activations not to blow up or collapse in magnitude.

**Q: One exception: Cohere's Command A and R+ use LayerNorm rather than RMSNorm. Any idea why?**
A (from Hashimoto): "I'm not quite sure why." This is a genuine open question -- it may relate to their specific training setup or the parallel layer architecture they use.

#### Advanced Questions for Students

1. **The 25% puzzle:** Normalization is 0.17% of FLOPs but 25% of runtime. Compute the arithmetic intensity (FLOPs per byte of memory accessed) for LayerNorm vs a matrix multiply. Why is it so different? (Hint: LayerNorm reads the entire vector, computes a scalar, then reads it again to normalize. The ratio of compute to data movement is terrible.)

2. **Would you expect the RMSNorm advantage to grow or shrink on future hardware?** Consider trends in compute-to-memory-bandwidth ratios on GPUs (the "arithmetic intensity gap"). What happens as GPUs get more FLOPS but memory bandwidth scales more slowly?

3. **RMSNorm removes the bias $\beta$ (shift parameter). Could this hurt for certain tasks?** Think about cases where the optimal activation distribution is not zero-centered.

### 1.3 Dropping Bias Terms (Universal trend)

Most modern transformers have **no bias terms** in linear layers or LayerNorm.

Original: $FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$
Modern: $FFN(x) = \sigma(xW_1)W_2$

**Reasons:**
1. Same memory/data movement argument as RMSNorm -- fewer parameters to load
2. **Optimization stability** -- empirically, dropping bias terms stabilizes training of very large networks (the mechanism isn't fully understood, but the effect is clear)

#### Teaching Intuition

Bias terms seem harmless -- they're just adding a constant. But think about what happens across many layers. Each bias adds an offset to the activation distribution. Across 80+ layers, these offsets accumulate in the residual stream. Now the model has to "fight" these accumulated biases just to keep activations in a reasonable range. Removing biases means the network only does linear transformations (matrix multiplies) and nonlinearities -- there's nothing pulling activations in arbitrary directions.

Also connect this to RMSNorm: if you've already removed the $\beta$ shift from normalization, keeping bias terms in linear layers reintroduces the same kind of shift you just removed. The whole trend is: **let the model learn through matrix multiplies alone, don't add unnecessary constant offsets.**

#### Advanced Questions for Students

1. **Parameter efficiency:** A bias vector in a $d_{model}$-dimensional linear layer adds $d_{model}$ parameters. For a model with $d_{model} = 4096$ and, say, 6 linear layers per block across 32 blocks, how many bias parameters is that? What fraction of total parameters? Is the memory argument actually compelling, or is something else going on?

2. **Bias and initialization:** At initialization, a linear layer without bias computes $y = Wx$ where $W$ is typically initialized with small random values. Adding a bias $b$ (often initialized to zero) shouldn't change anything at init. So why would removing bias help *stability*? (This remains an open question -- discuss hypotheses.)

### 1.4 Activations & Gated Linear Units (Strong trend toward SwiGLU/GeGLU)

**Evolution of activations:**

| Activation | Formula | Notable Models |
|---|---|---|
| ReLU | $FF(x) = \max(0, xW_1)W_2$ | Original transformer, T5, Gopher, OPT |
| GeLU | $FF(x) = GELU(xW_1)W_2$ where $GELU(x) = x\Phi(x)$ | GPT-1/2/3, GPT-J, BLOOM |
| SwiGLU | $FF(x) = (Swish(xW) \otimes xV)W_2$ | LLaMA 1/2/3, PaLM, Mistral, *most post-2023* |
| GeGLU | $FF(x) = (GELU(xW) \otimes xV)W_2$ | T5 v1.1, mT5, Phi3, Gemma 2/3 |

**What makes GLUs different:** Instead of just applying a nonlinearity, GLUs add a **gating mechanism**. The hidden representation is element-wise multiplied by a learned linear projection $xV$. This gate controls information flow through the MLP.

$$\text{Standard:} \quad \sigma(xW_1) \rightarrow \sigma(xW_1) \otimes (xV) \quad \text{(gated)}$$

**The extra parameter V** means GLU models have 3 weight matrices (W, V, W2) instead of 2. To keep parameter count matched, the hidden dimension $d_{ff}$ is scaled down by 2/3.

**Evidence GLUs work (Shazeer 2020, Narang et al 2020):** Consistently better performance across multiple benchmarks. GLU variants (GeGLU, SwiGLU, ReGLU) all outperform non-gated counterparts.

**Important caveat:** GLUs aren't *necessary* for good models. GPT-3 (GeLU), Nemotron 340B (Squared ReLU), and Falcon 2 11B (ReLU) are all strong models without gating. But evidence consistently favors GLUs.

**Q&A insight -- does the non-monotonic region of Swish/GeLU cause problems?** Hashimoto: In practice no, because optimization uses high learning rates with momentum, so activations don't converge to the zero point. Also, the FLOP cost of these fancier activations is negligible compared to the matrix multiplies; what matters is memory pressure, which is identical.

#### Teaching Intuition

Think of the MLP as a **two-step process**: (1) project into a high-dimensional space, apply nonlinearity, (2) project back. The gating mechanism adds a **learned filter** at step 1 -- the model gets to decide, for each hidden dimension, how much signal to let through. Without gating, the nonlinearity is a fixed function of the input. With gating, it's *input-dependent* -- the gate $xV$ learns which features are relevant for the current input and suppresses others.

A good analogy: imagine a photo editor. The standard MLP applies a fixed filter (contrast, brightness). The gated MLP first analyzes the photo to decide *which* filter to apply and how strongly -- it's adaptive. The $xV$ projection is the "analyzer" and the element-wise multiplication is the "adaptive filtering."

**Why the 2/3 scaling?** Walk through the parameter count. Standard MLP: $W_1 \in \mathbb{R}^{d \times d_{ff}}$ + $W_2 \in \mathbb{R}^{d_{ff} \times d}$ = $2 \cdot d \cdot d_{ff}$ params. Gated MLP: $W \in \mathbb{R}^{d \times d_{ff}}$ + $V \in \mathbb{R}^{d \times d_{ff}}$ + $W_2 \in \mathbb{R}^{d_{ff} \times d}$ = $3 \cdot d \cdot d_{ff}$ params. To match: set $d_{ff}^{gated} = \frac{2}{3} d_{ff}^{standard} = \frac{2}{3} \cdot 4d = \frac{8}{3}d$.

#### Student Questions (from the CS336 lecture)

**Q: ReLU is easily differentiable. GeLU/Swish involve the CDF of a Gaussian -- does that slow things down?**
A (from another student in the audience): "What really matters is memory pressure, and it's the exact same because you're reading the same number of elements." The extra arithmetic is negligible; the bottleneck is loading and storing the activation tensor, which is the same size regardless of the nonlinearity.

**Q: Below a certain negative value, Swish and GeLU are not monotonically increasing -- they actually decrease. Doesn't this break gradient descent?**
A: Intuitively you might think activations could get "trapped" in this negative region. But in practice, with high learning rates and momentum, activations are being pushed around aggressively. They don't converge to the problematic zero point. The small negative bump has negligible effect on training dynamics.

#### Advanced Questions for Students

1. **Parameter count exercise:** Prove that setting $d_{ff} = \frac{8}{3} d_{model}$ for a gated MLP gives exactly the same number of parameters as $d_{ff} = 4 \cdot d_{model}$ for a standard MLP. Then: do the FLOPs also match? (Careful: GLU has an extra element-wise multiply.)

2. **Why gating works:** Gating appears in LSTMs, GRUs, highway networks, and now GLU-style MLPs. What's the common principle? Can you formulate a general hypothesis for why learned multiplicative interactions help? (Consider: information routing, gradient flow, conditional computation.)

3. **Squared ReLU:** Nemotron 340B uses $\text{ReLU}^2(x) = (\max(0, x))^2$. This has even sparser activations than ReLU (smoother around zero, but kills more values). Why might sparsity in activations be beneficial? Could this relate to the "superposition hypothesis" from mechanistic interpretability?

4. **Design exercise:** You're building a 7B parameter model. You've decided on SwiGLU. Your $d_{model} = 4096$. Calculate: (a) What should $d_{ff}$ be? (b) Total MLP parameters per layer. (c) If you used standard ReLU MLP instead, what would $d_{ff}$ be for the same parameter count?

### 1.5 Serial vs Parallel Layers (Minor variation)

**Serial (standard):** Attention first, then MLP sequentially.
$$y = x + MLP(LN(x + Attention(LN(x))))$$

**Parallel:** Attention and MLP computed simultaneously, results added.
$$y = x + MLP(LN(x)) + Attention(LN(x))$$

**Benefits of parallel:**
- ~15% faster training (PaLM paper) because MLP and attention matrix multiplies can be fused
- LayerNorm can be shared between the two branches
- Quality neutral at large scale (small degradation at 8B, none at 62B per PaLM)

**Models using parallel:** GPT-J (pioneer), PaLM, GPT-NeoX, Cohere Command A/R+, Falcon 2 11B

**Current status:** Most recent models still use serial. Parallel hasn't been widely adopted despite the compute win, possibly because the quality tradeoff isn't worth it at smaller scales.

#### Teaching Intuition

In a serial block, the MLP gets to *see the output of attention* -- it can refine what attention computed. In a parallel block, MLP and attention work *independently* on the same input and their outputs are simply summed. This is like the difference between two people working *sequentially* (one reviews and improves the other's work) vs *in parallel* (both work independently, results are merged).

The PaLM team's finding is telling: at 8B scale, parallel hurts slightly. At 62B, no difference. Their hypothesis: at large enough scale, the model has so much capacity that the loss from not composing attention + MLP is negligible. The compute savings from fusing operations outweigh the small expressiveness loss.

#### Student Questions (from the CS336 lecture)

**Q: Is serial more efficient than parallel?**
A: Actually the reverse -- parallel is more compute-efficient (15% faster training). The concern is about expressiveness: serial composes two computations, while parallel merely adds them. You might expect serial to be more expressive. The tradeoff is: parallel gains systems efficiency but potentially loses expressiveness.

#### Advanced Questions for Students

1. **Expressiveness argument:** The serial formulation computes $MLP(LN(x + Attn(LN(x))))$ -- the MLP can be a function of the attention output. The parallel computes $MLP(LN(x)) + Attn(LN(x))$ -- both branches see only $x$. Give a concrete example of a computation that serial can express but parallel cannot. (Hint: think about attention selecting information that the MLP then needs to process.)

2. **Fusion opportunity:** Explain specifically which matrix multiplies can be fused in the parallel formulation but not in serial. Draw the computation graph for both and identify the kernel fusion boundaries.

3. **Why hasn't parallel won?** Despite the 15% training speedup, most 2024-2025 models use serial. Hypothesize: is this conservatism (copying LLaMA), or is there a real quality concern? How would you design an experiment to settle this?

### 1.6 Position Embeddings (Converged to RoPE)

This is the area that saw the most exploration in 2017-2022, but has now largely converged.

**Evolution:**

| Type | How it works | Models |
|---|---|---|
| **Sinusoidal** | Add fixed sin/cos to embedding | Original Transformer |
| **Absolute (learned)** | Add learned position vector $u_i$ to embedding | GPT-1/2/3, OPT |
| **Relative** | Add learned bias to attention scores | T5, Gopher, Chinchilla |
| **ALiBi** | Linear attention bias | BLOOM |
| **NoPE** | No position embedding at all | SmolLM3, Kimi Linear |
| **RoPE** | Rotate query/key vectors | GPT-J, PaLM, LLaMA, **all 2024+ models** |

**RoPE's key idea:**

We want attention scores to depend only on *relative* position $(i - j)$, not absolute positions. Mathematically, find $f(x, i)$ such that:
$$\langle f(x, i), f(y, j) \rangle = g(x, y, i-j)$$

**Why existing methods fail this:**
- Sinusoidal: Cross-terms leak absolute position info
- Absolute: Obviously not relative
- T5 relative: Modifies the attention computation, but the result isn't a standard inner product

**RoPE's solution:** Rotations preserve inner products. If you rotate each embedding vector by an angle proportional to its position, the inner product between any two vectors depends only on their angular *difference* (= relative position).

**Implementation:** Pair up dimensions (d/2 pairs), apply 2D rotations with different frequencies per pair (analogous to sinusoidal frequencies). The rotation matrix is block-diagonal with 2x2 rotation blocks:

$$R_{\Theta,m} = \text{block-diag}(\text{Rot}(m\theta_1), \text{Rot}(m\theta_2), \ldots, \text{Rot}(m\theta_{d/2}))$$

**Critical difference from sinusoidal:** RoPE is **multiplicative** (applied at each attention layer to Q and K), not additive (at input). This is essential for enforcing position invariance at every attention operation.

**Why RoPE won:** All 19 papers Hashimoto surveyed from 2024-2025 use RoPE. Beyond the theoretical elegance, RoPE has proven practically effective and has spawned many context-length extension algorithms (NTK-aware scaling, YaRN, etc.).

#### Teaching Intuition

**The "we know" example from the lecture is gold for teaching.** Walk through it step by step:

1. Start with two word vectors: $\vec{we}$ and $\vec{know}$ (arrows in 2D for visualization).
2. Sentence "we know that" -- "we" is at position 0, "know" at position 1. Rotate "we" by 0 radians, "know" by $\theta$ radians. The angle between them is $\theta$.
3. Sentence "of course we know" -- "we" is at position 2, "know" at position 3. Rotate "we" by $2\theta$, "know" by $3\theta$. The angle between them is still $\theta$.
4. **Key insight:** The inner product $\langle R_{2\theta}\vec{we}, R_{3\theta}\vec{know} \rangle = \langle R_0\vec{we}, R_\theta\vec{know} \rangle$ because rotations preserve inner products and only the *difference* in rotation angles matters.

This is the entire idea. Everything else is implementation details (how to handle high dimensions, how to pick $\theta$s).

**The high-dimensional extension:** We can't easily rotate in $d$-dimensional space (too many degrees of freedom). RoPE's clever trick: pair up consecutive dimensions and rotate each pair independently. This gives $d/2$ independent 2D rotations, each with its own frequency -- exactly analogous to how sinusoidal embeddings have different frequencies for different dimension pairs.

**Why it's multiplicative, not additive:** Draw the code path. Absolute embeddings: add position vector at the input, once. RoPE: multiply Q and K by rotation matrix at *every* attention layer. This is necessary because if you only add at the bottom, the position signal gets diluted by subsequent transformations. By applying at each attention operation, you guarantee position information is fresh.

#### Student Questions (from the CS336 lecture)

**Q: Are the rotation angles ($\theta$) hyperparameters or learned?**
A: Neither -- they follow a fixed schedule (like sinusoidal embeddings). Different dimension pairs rotate at different speeds: some fast (capturing local/nearby token relationships), some slow (capturing long-range relationships). The thetas are not learned because rotating is just a matrix multiply with fixed coefficients -- if you were learning thetas, you'd need to differentiate through trig functions, which could cause issues.

**Q: Is the rate of rotation consistent across models?**
A: There's some variation in the $\theta$ schedule across models. The original RoPE uses $\theta_i = 10000^{-2i/d}$. Later work (NTK-aware RoPE, YaRN) modifies these to enable context length extension.

**Q: Do the rotations create difficulty with training?**
A: No, because a rotation is just a fixed matrix multiply (since $\theta$s and positions $m$ are fixed during forward pass). It's no different from any other linear transformation. Gradients flow through it normally. If you were *learning* the rotation parameters, then differentiating through trig functions could be problematic, but you're not.

#### Advanced Questions for Students

1. **Prove the relative position property:** Given $f(x, m) = R_m W x$ where $R_m$ is the RoPE rotation matrix, show that $\langle f(x, m), f(y, n) \rangle = g(x, y, m-n)$ for some function $g$. Where does the proof break if we use additive position embeddings instead?

2. **Context length extrapolation:** RoPE was originally trained on sequences of length $L$ with $\theta_i = 10000^{-2i/d}$. At inference, we want to use length $4L$. What goes wrong? The rotation angles $m\theta_i$ for positions $m > L$ were never seen during training. How do NTK-aware scaling and YaRN address this? (Hint: they modify the base frequency or interpolate positions.)

3. **RoPE vs ALiBi:** ALiBi (Attention with Linear Biases) achieves relative position encoding by subtracting a linear penalty from attention scores: $\text{score}_{ij} = q_i \cdot k_j - m \cdot |i-j|$, where $m$ is a per-head slope. Compare the expressiveness: RoPE modulates the *content-based* similarity via rotation, while ALiBi adds a *content-independent* distance penalty. In what scenarios might one be better than the other?

4. **NoPE layers (No Position Embedding):** In the Command A / LLaMA 4 architecture, some layers use full attention with *no* position embedding at all. How can a transformer layer function without position information? What does this layer compute? (It can attend to all positions equally based on content only -- think of it as a pure "what's similar to me?" lookup.)

---

## Part 2: Hyperparameters

### 2.1 FFN Dimension Ratio: $d_{ff} = 4 \cdot d_{model}$ (or 8/3 for GLUs)

**The rule:** Almost universally, the FFN hidden dimension is 4x the model dimension.

For GLU variants (which have an extra parameter matrix V), scale down by 2/3 to maintain parameter parity: $d_{ff} = \frac{8}{3} d_{model} \approx 2.67 \cdot d_{model}$

**Models following the 8/3 rule:** LLaMA 70B (2.68), Qwen 14B (2.67), DeepSeek 67B (2.68), Yi 34B (2.85)
**Slightly larger:** PaLM (4.0), Mistral 7B (3.5), LLaMA-2 70B (3.5)

**Notable exception -- T5 11B:** Uses $d_{ff} = 65,536$ with $d_{model} = 1024$ -- a **64x multiplier**. Rationale from the paper: larger FFN means larger matrix multiplies, which TPUs handle more efficiently. However, T5 v1.1 walked this back to a 2.5 multiplier with GeGLU and got a better model.

**Empirical evidence (Kaplan et al 2020):** There's a wide basin between ratios 1-10 where performance is near-optimal. The 4x default sits comfortably within this basin.

#### Teaching Intuition

The MLP does two things: **project up** to a higher-dimensional space (where nonlinearities can carve out complex decision boundaries), then **project back** to the model dimension. The ratio controls how much "room" the model has for intermediate computation. At 4x, you're saying: "give the MLP 4 times as many hidden neurons as its input dimension to do its computation."

Why does a wide basin exist (1-10x all work)? Because the model can compensate: if $d_{ff}$ is smaller, each MLP does less work, but you have many layers, so the total computation budget is still large. It's only at extreme ratios (>10x) where you're spending parameters very inefficiently that things degrade.

**The T5 story is great for teaching:** "LLM training is a game of copying hyperparameters from other people" (Hashimoto). T5 was bold enough to try 64x -- and it worked! But then T5 v1.1 quietly walked it back to 2.5x and got a *better* model. The lesson: you *can* break conventions, but the conventions exist for reason.

#### Advanced Questions for Students

1. **Where do the MLP parameters live?** For a standard model with $d_{ff} = 4 d_{model}$, what fraction of total layer parameters are in the MLP vs attention? Compute for both standard (2 matrices: $d \times 4d$ and $4d \times d$) and gated (3 matrices with $d_{ff} = 8/3 \cdot d$). Which dominates?

2. **The T5 rationale:** T5 chose 64x because "modern accelerators are most efficient for large dense matrix multiplications." Explain this argument in terms of arithmetic intensity. When does making one dimension very large help GPU utilization? When does it become counterproductive?

### 2.2 Head Dimension Ratio: $d_{head} \times n_{heads} = d_{model}$

The standard practice is to split the model dimension evenly across heads, so each head gets $d_{model}/n_{heads}$ dimensions. This means adding more heads doesn't increase compute cost.

| Model | Num heads | Head dim | Model dim | Ratio |
|---|---|---|---|---|
| GPT-3 | 96 | 128 | 12288 | 1 |
| T5 | 128 | 128 | 1024 | **16** |
| LLaMA-2 | 64 | 128 | 8192 | 1 |
| PaLM | 48 | 258 | 18432 | 1.48 |

T5 is again the outlier (ratio of 16). Bhojanapalli et al (2020) argued theoretically against the 1:1 ratio (low-rank bottleneck per head), but in practice the bottleneck doesn't seem to bite.

#### Teaching Intuition

Multi-head attention is a clever trick: instead of one big attention computation, we run $h$ smaller ones in parallel. The key insight is that this **doesn't cost extra compute**. We still compute one big $XQ$ multiplication ($d \times d$ parameters) and then *reshape* the result into $h$ heads, each with $d/h$ dimensions. The total parameter count and FLOPs are the same whether we use 1 head or 96 heads.

But each head only has $d/h$ dimensions to work with. If $d/h$ is very small (say, 8 or 16), each head can only represent very low-rank attention patterns. The theoretical concern is: with 128 heads and $d = 4096$, each head only has 32 dimensions -- is that enough to compute meaningful attention patterns?

In practice: yes, it seems to be. Models with the 1:1 ratio work great. The theoretical low-rank bottleneck doesn't seem to matter empirically.

#### Advanced Questions for Students

1. **Rank analysis:** Each attention head computes $\text{softmax}(Q_h K_h^T / \sqrt{d_h})$, where $Q_h, K_h \in \mathbb{R}^{n \times d_h}$. What is the maximum rank of $Q_h K_h^T$? For $d_h = 128$ and $n = 4096$, is the attention matrix rank-limited? Does this matter?

2. **Breaking the 1:1 ratio:** With GQA, we have fewer KV heads than query heads. This means the KV projection is $d_{model} \rightarrow n_{kv\_heads} \times d_{head}$ where $n_{kv\_heads} < n_{q\_heads}$. The 1:1 ratio now only applies to Q. What are the implications for the attention patterns' expressiveness?

### 2.3 Aspect Ratio: $d_{model} / n_{layers} \approx 100-200$

How deep vs wide should the model be?

| Model | $d_{model}/n_{layers}$ |
|---|---|
| GPT-3/OPT/Mistral/Qwen | 128 |
| LLaMA/LLaMA-2/Chinchilla | 102 |
| PaLM (540B) | 156 |
| T5 (11B) | 43 (outlier) |
| GPT-2 | 33 (outlier) |

**Sweet spot is 100-200.** Evidence from Kaplan et al (2020) shows this is optimal across scales (50M, 274M, 1.5B parameters).

**Systems consideration:** Very deep models are harder to parallelize (pipeline parallelism has latency constraints), while very wide models can use tensor parallelism (requires fast networking). Networking constraints may drive depth/width choices.

**Depth vs width for downstream tasks:** Tay et al (2021) found that while upstream loss depends mainly on parameter count, downstream accuracy (e.g., SuperGLUE) may favor deeper models at equal FLOPs. This isn't fully settled.

#### Teaching Intuition

Aspect ratio is the "shape" of your model -- tall and thin vs short and wide. Think of it like a building: more floors (layers) means more sequential processing, while wider floors (larger $d_{model}$) means more parallel processing per step.

**Why ~128 hidden dims per layer?** No one has a clean theoretical answer. It's empirically derived: Kaplan et al (2020) show that across 3 orders of magnitude in model size, the optimal aspect ratio stays roughly constant. This is remarkable -- it means when you double your parameter budget, you should make the model both deeper AND wider in roughly equal proportion.

**The systems angle is critical for teaching:** A very deep model (many layers, small $d_{model}$) is hard to parallelize across GPUs because layers are sequential -- GPU 1 must finish layer 1 before GPU 2 can start layer 2 (pipeline parallelism). A very wide model (few layers, huge $d_{model}$) can split each layer's matrix multiply across many GPUs (tensor parallelism), but this requires extremely fast inter-GPU communication (NVLink, not just PCIe). Real clusters have specific network topologies that constrain what's feasible. So the "optimal" aspect ratio isn't just about model quality -- it's about what your hardware can efficiently execute.

#### Advanced Questions for Students

1. **Scaling exercise:** You have a 7B parameter budget. Using the consensus ratios ($d_{ff} = 8/3 \cdot d_{model}$, SwiGLU, $d_{head} = 128$, aspect ratio ~128), work out: $d_{model}$, $n_{layers}$, $n_{heads}$, $d_{ff}$. Verify total parameter count. Compare your answer to LLaMA-2 7B's actual architecture.

2. **Pipeline vs tensor parallelism tradeoff:** You have 8 GPUs connected with NVLink (fast) within a node, and 4 nodes connected with InfiniBand (slower). You're training a model with 80 layers and $d_{model} = 8192$. How would you distribute the model? What changes if the model had 160 layers and $d_{model} = 4096$ (same parameter count)?

3. **The depth-downstream mystery:** Tay et al (2021) showed deeper models are better for downstream tasks even when upstream loss is the same. Why might depth help with fine-tuning? (Hypothesize: deeper networks compose more functions, enabling more abstract representations that transfer better. Or: deeper networks have more "slots" for task-specific adaptation during fine-tuning.)

### 2.4 Vocabulary Size

**Monolingual models:** 30-50K tokens (Original Transformer: 37K, GPT-2/3: 50K, LLaMA: 32K)
**Multilingual/Production:** 100-250K tokens (GPT-4: 100K, PaLM: 256K, Qwen: 152K, Command A: 255K)

Trend is upward as models serve more diverse users and languages. Larger vocabularies particularly help low-resource languages by packing them into fewer tokens (reducing inference cost for those users).

#### Teaching Intuition

Vocabulary size involves a fundamental tradeoff: **compression vs coverage**. Larger vocab = each token carries more information (fewer tokens per sentence = faster inference, cheaper API calls), but also = larger embedding matrix (more parameters, slower softmax). Smaller vocab = model sees more tokens per concept (more compositional, potentially better generalization), but longer sequences (slower, more expensive).

For production multilingual systems, the equation tilts heavily toward larger vocabs. Cohere's argument: with a large, multilingual tokenizer, a Hindi sentence might use 50 tokens instead of 200. That's 4x cheaper inference for Hindi users -- a direct business and equity argument.

#### Student Questions (from the CS336 lecture)

**Q: Do multilingual vocabularies actually improve performance in one language (e.g., English)?**
A: For high-resource languages like English, the impact is small -- you can get by with 32K tokens. The value of larger vocabularies is primarily for lower-resource languages, where they dramatically reduce token count and inference cost.

#### Advanced Questions for Students

1. **Embedding matrix cost:** The embedding matrix is $V \times d_{model}$ parameters. For $V = 256000$ and $d = 4096$, how large is this in GB (at bf16)? What fraction of a 7B model's total parameters? At what point does the embedding matrix become the dominant cost?

2. **The efficiency argument:** Larger vocab → fewer tokens per document → fewer forward passes at inference. But also: larger vocab → larger softmax computation at every step. When does the trade off favor larger vocabs? Derive a rough expression for total inference cost as a function of vocab size, average compression ratio, and model size.

3. **Tokenizer-model co-design:** Should vocabulary size decisions be made independently of architecture decisions? Consider: a model with $d_{model} = 1024$ may not have enough representational capacity to embed 256K distinct tokens meaningfully. How would you design an experiment to find the optimal vocab size as a function of model size?

### 2.5 Regularization: Weight Decay Yes, Dropout No

**Dropout has gone out of fashion** for pretraining. Newer models (post-2022) mostly don't use it. Rationale: with trillions of tokens and single-epoch training, there's no overfitting concern.

**Weight decay persists** (typically 0.1), but **not for regularization**. This is counterintuitive.

**Why weight decay helps (Andriushchenko et al 2023):**
- Train/val gap is identical regardless of weight decay amount -- it's not controlling overfitting
- Weight decay interacts with learning rate schedules: with cosine LR decay, high weight decay models start slower but accelerate dramatically near the end of training
- The result is **lower training loss** (which equals better val loss since there's no overfitting)

#### Teaching Intuition

This is a great "intuition-busting" topic. Walk students through the reasoning:

**Step 1 (the puzzle):** You have trillions of tokens, billions of parameters, and you train for one epoch. There's zero overfitting risk. Why would you regularize?

**Step 2 (the observation):** Look at Andriushchenko's plots. Train loss vs val loss: identical lines regardless of weight decay. It's not preventing overfitting. But training loss *itself* is lower with weight decay.

**Step 3 (the mechanism):** Weight decay shrinks weights. With a cosine learning rate schedule, the learning rate starts high and decays to near-zero. Early in training (high LR), weight decay is fighting against the large updates, slowing learning. Late in training (low LR), the weights have been kept small by weight decay, and the model can now make very precise, fine-grained adjustments. It's as if weight decay creates a "compressed spring" that releases energy at the end of training.

**The punchline:** Weight decay in LLM pretraining is not regularization in the classical sense. It's an **optimization trick** that improves training dynamics by interacting with the learning rate schedule.

#### Student Questions (from the CS336 lecture)

**Q: Why did dropout go out of fashion?**
A: There's no evidence it helps training loss, and since there's no overfitting problem (single epoch over vast data), there's no regularization need either. It also complicates distributed training (need to synchronize dropout masks or deal with variance).

**Q: If weight decay doesn't affect val loss relative to train loss, why do we care about the training dynamics?**
A: Because the game is minimizing training loss -- that's the objective. Weight decay somehow gets us to lower training losses (which are also lower val losses, since the gap is constant). The surprising part is that it achieves this through optimization dynamics, not regularization.

#### Advanced Questions for Students

1. **AdamW vs L2 regularization:** In standard SGD, weight decay and L2 regularization are equivalent. In Adam, they're not (this is why AdamW exists). Explain the difference. Why does this distinction matter for LLM training? (Hint: Adam's adaptive learning rates interact differently with the penalty term.)

2. **The cosine schedule interaction:** Sketch what happens to the effective learning rate (LR × gradient magnitude) with and without weight decay under a cosine schedule. Why does weight decay create a "spring-loading" effect? Could you achieve the same effect with a different LR schedule and no weight decay?

3. **Dropout in fine-tuning:** Even though dropout isn't used in pretraining, it's still common in fine-tuning. Why might the calculus be different? (Fine-tuning: small dataset, many epochs, real overfitting risk. The arguments against dropout vanish.)

---

## Part 3: Stability Tricks (New in 2024-2025)

This is the area Hashimoto identified as having the most new development in the past year. As models get larger and train longer, stability becomes critical.

### 3.1 The Problem: Softmaxes

Two softmaxes in a transformer are potential instability sources:
1. **Output softmax** (final logit → probability conversion)
2. **Attention softmax** (QK^T → attention weights)

Both involve exponentials (can overflow) and division (can be zero).

#### Teaching Intuition

Show students the OLMo training curves: the blue curve (unstable) has gradient norm spikes everywhere -- imagine paying $10M to train a model and watching it diverge at step 400K. The orange curve (stable) is smooth. The difference? A few small architectural interventions. This motivates *why* stability tricks matter: they're not about making models better, they're about making sure your $10M training run doesn't crash and burn.

**Why softmaxes are dangerous:** $\text{softmax}(x)_i = e^{x_i} / \sum_j e^{x_j}$. If any $x_i$ becomes very large (say, 1000), $e^{1000}$ overflows float32/bfloat16. If all $x_i$ are very negative, $\sum_j e^{x_j} \approx 0$, and you divide by zero. These are the two failure modes: **overflow** and **underflow**. Both create NaN/Inf gradients that corrupt the entire training.

### 3.2 Z-Loss (Output Softmax Stability)

**Idea (Devlin 2014, popularized by PaLM 2022):** Add an auxiliary loss to keep the softmax normalizer $Z(x) = \sum_{r'} e^{U_{r'}(x)}$ close to 1:

$$L = \sum_i [\log(P(x_i)) - \alpha \cdot \log^2(Z(x_i))]$$

When $\log(Z) \approx 0$ (i.e., $Z \approx 1$), the softmax computation simplifies to just $U_r(x)$ -- no exponentials or log-sum-exp, which is numerically clean.

**Adopted by:** PaLM (pioneer), Baichuan 2, DCLM, OLMo 2

#### Teaching Intuition

Walk through the math slowly. The log-softmax is $\log P(x) = U_r(x) - \log Z(x)$. If $Z(x)$ is well-behaved (close to 1), then $\log Z \approx 0$ and $\log P \approx U_r$ -- you're just reading off the logit directly. No exponentials, no division. Clean gradients.

The Z-loss penalty $\alpha \cdot \log^2 Z$ is a gentle nudge: "please keep $Z$ close to 1." It doesn't force it -- the model is free to have $Z \neq 1$ if that helps the main loss -- but there's a cost for letting $Z$ drift far from 1.

**Analogy:** It's like putting a leash on a dog. The dog (the model) can go wherever it wants, but if it pulls too far (Z gets large), the leash (Z-loss) tugs back. The coefficient $\alpha = 10^{-4}$ makes it a very loose leash -- only extreme deviations get penalized.

### 3.3 QK-Norm (Attention Softmax Stability)

**Idea:** Apply LayerNorm to queries and keys *before* computing attention scores. This bounds the inputs to the softmax, preventing extreme values.

**Origin:** Vision/multimodal community (Dehgani 2023 for large ViTs, then Chameleon, IDEFICS from Hugging Face). The innovation then migrated to pure text LLMs.

**Adopted by:** Gemma 2, DCLM, OLMo 2

**Meta-lesson:** LayerNorm is strikingly effective as a stability tool. It's been added at pre-norm position, post-norm-outside-residual position, and now inside attention for QK normalization.

### 3.4 Logit Soft-Capping (Less Common)

**Idea (Gemma 2):** Cap attention logits using tanh:
$$\text{logits} \leftarrow \text{soft\_cap} \cdot \tanh(\text{logits} / \text{soft\_cap})$$

Prevents logits from exceeding $\pm$soft_cap. Gemma 2 uses soft_cap=50 for attention, 30 for final layer.

**Mixed evidence:** Nvidia ablation showed soft-capping slightly *hurts* perplexity (11.24 vs 11.19 baseline), while QK-norm improves it (10.84-10.85). So QK-norm appears to be the better stability intervention.

#### Student Questions (from the CS336 lecture)

**Q: For QK-norm, the LayerNorm is applied during training. Is it kept at inference time?**
A: Yes, absolutely. The LayerNorm has learned parameters ($\gamma$) that the model depends on. Removing it at inference would be like removing a layer from the network -- the model would produce garbage because it was trained to expect normalized QK inputs.

#### Advanced Questions for Students

1. **Z-loss vs QK-norm:** Both stabilize softmaxes, but they work differently. Z-loss is a *loss function* modification (the forward pass is unchanged). QK-norm is an *architecture* modification (adds parameters and computation). What are the tradeoffs? When might you prefer one over the other?

2. **Where do the instabilities come from?** In the attention softmax, the inputs are $QK^T / \sqrt{d_k}$. If $Q$ and $K$ have entries that grow during training (activation drift), then $QK^T$ can have entries that are very large. Trace through *why* activations might grow during training. (Hint: residual stream accumulation, lack of normalization, training dynamics.)

3. **The LayerNorm meta-lesson:** Hashimoto's joke: "stack more layer norms." We've now seen LayerNorm used at: pre-norm position, double-norm position, QK-norm position. Is there a principled theory for *where* normalization helps? Or is it just "throw a norm at any unstable computation and it usually helps"? What are the costs of excessive normalization?

4. **Soft-capping as an alternative to FlashAttention:** The standard numerical trick for stable softmax is to subtract $\max(x)$ before exponentiating. FlashAttention handles this automatically. Logit soft-capping with tanh prevents large values altogether. If you're already using FlashAttention with the max-subtraction trick, does soft-capping provide additional benefit? Why might it *hurt* perplexity?

---

## Part 4: Attention Variants

### 4.1 GQA / MQA (Inference Optimization)

**Problem:** During autoregressive generation, the KV cache creates terrible arithmetic intensity. At training time, attention has high arithmetic intensity $O((1/k + 1/bn)^{-1})$. At inference (incremental decoding), it drops to $O((n/d + 1/b)^{-1})$ -- the $n/d$ term is problematic because you want long sequences ($n$ large) but can't easily increase $d$.

**Multi-Query Attention (MQA):** Keep multiple query heads but use a **single shared** key and value head. This dramatically reduces KV cache size and memory movement.

**Grouped-Query Attention (GQA):** A middle ground -- share K/V across groups of query heads (e.g., 8 query heads share 2 KV heads). Provides a knob to trade off expressiveness vs inference efficiency.

**Evidence:** MQA has a small PPL hit (30.2 vs 29.9 multi-head; Shazeer 2019). GQA shows low to no quality hit while providing major inference speedups (Ainslie 2023).

#### Teaching Intuition

**The KV cache explanation is essential.** Walk through it step-by-step:

1. At training time, we process the entire sequence at once. We compute $Q, K, V$ for all positions simultaneously. The attention matrix multiply $QK^T$ is a fat matrix multiply -- GPUs love this.

2. At inference, we generate tokens one-at-a-time. For each new token, we compute only one new row of $Q$ (1 token). But we need to multiply it against *all* previous $K$'s (accumulated in the KV cache). This is a matrix-vector multiply (one row of Q against the full K matrix) -- terrible arithmetic intensity.

3. **The bottleneck isn't compute, it's memory.** We have to *load* the entire KV cache from GPU memory for every single token generation. For a model with 32 heads, 128 dims per head, and 100K context tokens, the KV cache is $2 \times 32 \times 128 \times 100000 \times 2$ bytes $\approx$ 1.6 GB *per layer*. For 80 layers: 128 GB. Loading this from HBM for every token is brutally slow.

4. **MQA's fix:** If we share K and V across all heads, the cache shrinks by a factor of $n_{heads}$ (e.g., 32x smaller). GQA with 4 KV groups gives an 8x reduction. Dramatically less memory to move around.

**Key insight to drive home:** GQA/MQA is purely about inference efficiency. It barely affects training. The architecture decision is driven by *deployment* economics -- how many tokens per second can you serve, at what cost per query.

#### Advanced Questions for Students

1. **KV cache size calculation:** For LLaMA-2 70B (80 layers, 64 heads, $d_{head} = 128$, GQA with 8 KV groups), calculate the KV cache size in GB for a context of 128K tokens at bf16. How does this compare to the model weights size? At what context length does the KV cache exceed the model weights?

2. **The arithmetic intensity argument:** Derive the arithmetic intensity for standard multi-head attention at inference (one token at a time). Show that it's $O((n/d + 1/b)^{-1})$. Then derive it for MQA and show the improvement. At what batch size does MQA stop being beneficial? (i.e., when does the $1/b$ term dominate regardless.)

3. **Quality tradeoff:** MQA forces all query heads to attend using the same K and V. Conceptually, this means different attention heads can no longer look at different "aspects" of the keys and values -- they all share the same representation. Why doesn't this hurt more? (Hypothesize: the query projections already provide enough diversity, and the shared K/V still contains all the information.)

4. **Training GQA from scratch vs converting:** Ainslie et al (2023) showed you can convert a trained MHA model to GQA by mean-pooling the KV heads, then fine-tuning briefly. Why does this work? What information is lost in the mean-pooling, and why can fine-tuning recover it?

### 4.2 Sliding Window + Full Attention Interleaving (2025 State of the Art)

**The modern trick for long context:**
- Every Nth layer (e.g., every 4th) uses **full self-attention with no position embedding** (NoPE)
- Remaining layers use **sliding window attention with RoPE**

**Example (Cohere Command A):** Blocks 1-3 use SWA+RoPE, Block 4 uses Full+NoPE, repeat.

**Why this works:**
- **Short-range:** RoPE + sliding window handles local context efficiently
- **Long-range:** Full attention layers with no position embedding can extrapolate to arbitrary lengths (no position encoding to break down)
- **Systems:** Full attention only happens every N layers, controlling compute cost

**Models using this:** LLaMA 4, Gemma 3, Cohere Command A

#### Teaching Intuition

**The "interleaving" trick is brilliant and worth unpacking carefully:**

Think of the transformer as building up understanding layer by layer. In the SWA+Full interleaving:

- **SWA layers (with RoPE):** "Look at nearby tokens and understand local patterns." The sliding window means you only see, say, 4096 tokens around you. RoPE tells you exactly how far away each token is. These layers are cheap (sparse attention) and handle local syntax, word relationships, etc.

- **Full attention layers (with NoPE -- no position embedding):** "Now look at the entire document and find relevant information anywhere." No position embedding means the model doesn't know *where* things are -- just *what* they are. This is purely content-based retrieval. And because there's no position encoding, it doesn't break when the document is longer than anything seen in training.

**The key insight for context length:** The reason RoPE breaks at longer contexts is that the rotation angles $m\theta$ for large $m$ were never seen during training -- the model doesn't know how to handle them. By using NoPE for long-range attention, you sidestep this entirely. The full-attention layers don't care about position, so they work at any length.

**Analogy:** It's like reading a book with two strategies. Most of the time (SWA layers), you read carefully, knowing exactly which page and paragraph you're on (RoPE). Occasionally (full attention layers), you pause and think "where else in this entire book was this concept mentioned?" -- doing a pure content search without caring about page numbers (NoPE).

#### Advanced Questions for Students

1. **Design tradeoff:** If every 4th layer uses full attention (quadratic in sequence length) and the other 3 use sliding window (linear), what's the overall complexity as a function of sequence length $n$ and window size $w$? For a model with 32 layers and $w = 4096$, at what context length does the full attention cost dominate?

2. **NoPE expressiveness:** A full attention layer with no position embedding computes: $\text{softmax}(QK^T/\sqrt{d})V$. Position information is only available through what earlier layers have "written" into the residual stream. How does the model disambiguate two identical tokens at different positions? (Answer: earlier SWA+RoPE layers have already baked position-dependent information into the token representations.)

3. **Why not all-NoPE?** If NoPE enables infinite context, why not use it everywhere? Design an experiment showing where pure NoPE would fail. (Hint: word order matters in language -- "dog bites man" vs "man bites dog" need position information to distinguish.)

4. **KV cache implications:** In the interleaved architecture, SWA layers only cache the last $w$ tokens. Full attention layers cache everything. Calculate the KV cache savings compared to full attention everywhere. For a 128K context with $w = 4096$ and every 4th layer being full attention, what's the reduction?

---

## The "LLaMA-like" Modern Consensus Architecture

Combining all the above, the modern default is:

| Component | Choice |
|---|---|
| Normalization | **RMSNorm, pre-norm** |
| Bias terms | **None** |
| Activation/FFN | **SwiGLU** with $d_{ff} \approx 2.67 \cdot d_{model}$ |
| Position embedding | **RoPE** (applied at each attention layer) |
| Layer structure | **Serial** (attention → MLP) |
| Attention | **GQA** for inference efficiency |
| Aspect ratio | $d_{model}/n_{layers} \approx 100-200$ |
| Head dim | $d_{head} \times n_{heads} = d_{model}$ |
| Regularization | Weight decay 0.1, no dropout |
| Stability | Z-loss + QK-norm |

---

## Key References

- **Pre/Post-Norm:** Xiong et al 2020 "On Layer Normalization in the Transformer Architecture"; Salazar & Nguyen 2019
- **RMSNorm:** Zhang & Sennrich 2019; Narang et al 2020
- **GLU/SwiGLU:** Shazeer 2020 "GLU Variants Improve Transformer"; Narang et al 2020
- **RoPE:** Su et al 2021 "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- **Parallel Layers:** Wang & Komatsuzaki 2021 (GPT-J); Chowdhery et al 2022 (PaLM)
- **Aspect Ratio/Scaling:** Kaplan et al 2020 "Scaling Laws for Neural Language Models"; Tay et al 2021
- **FLOPS vs Runtime:** Ivanov et al 2023
- **Weight Decay:** Andriushchenko et al 2023
- **Z-Loss:** Devlin 2014; Chowdhery et al 2022 (PaLM)
- **QK-Norm:** Dehgani 2023; Henry et al (OLMo 2)
- **Logit Soft-Capping:** Gemma 2 Technical Report
- **GQA/MQA:** Shazeer 2019; Ainslie et al 2023
- **Sliding Window:** Child et al 2019; Jiang et al 2023 (Mistral)

---

## Cross-Cutting Themes for Discussion

These are higher-level discussion topics that span multiple sections of the lecture. Good for class discussion or exam questions.

### Theme 1: Theory vs Practice in Architecture Design

Almost no architecture choice in modern LLMs has a clean theoretical justification. Pre-norm: gradient arguments are hand-wavy. RMSNorm: the real reason is data movement, not math. GLUs: "they just work." RoPE: has nice math, but the practical success is what drove adoption.

**Discussion prompt:** "Is deep learning architecture design more like engineering or science? Are we doing principled design or glorified hyperparameter search? What would a more principled approach look like?"

### Theme 2: The Role of Systems in Architecture

A recurring theme: architecture choices are increasingly driven by hardware constraints, not model quality.

- RMSNorm wins because of **memory bandwidth**, not mathematical superiority
- Parallel layers win because of **kernel fusion**, not expressiveness
- GQA/MQA exist because of **KV cache memory**, not training quality
- Aspect ratios are constrained by **parallelism strategies**, not optimal loss
- SWA+Full interleaving is designed for **inference efficiency**, not training loss

**Discussion prompt:** "As hardware changes (more compute, relatively less memory bandwidth), which architecture choices might change? What happens if we move to architectures optimized for, say, TPUs vs GPUs vs custom accelerators?"

### Theme 3: Convergent Evolution and Conservatism

Hashimoto's metaphor of "convergent evolution" is apt. The field has converged on a "LLaMA-like" template. But is this convergence because these choices are genuinely optimal, or because everyone is copying from the same successful models?

**Discussion prompt:** "T5 tried radical hyperparameters (64x FFN ratio, 16x head ratio) and it worked. Yet almost no one followed. Why? Is the field too conservative? How would you distinguish 'this is the optimal choice' from 'this is what everyone copies'?"

### Theme 4: What We Don't Know

Honest areas of ignorance from the lecture:
- Why exactly do bias terms hurt stability?
- Why does weight decay improve training loss through optimizer dynamics?
- Is there a principled theory for where normalization helps?
- What's the true optimal aspect ratio, and does it shift with scale?
- Why haven't parallel layers been more widely adopted despite clear compute wins?

**Discussion prompt:** "Pick one of these open questions. Design a set of experiments that would give a convincing answer. What scale would you need to run at? What baselines and controls?"

---

## Suggested Exercises for Students

### Exercise 1: Architecture Autopsy (20 min)
Give students a recent model paper (e.g., Qwen-2.5 or SmolLM technical report). Have them fill in a table:

| Component | This model's choice | "Standard" choice | Same or different? |
|---|---|---|---|
| Normalization type | ? | RMSNorm | |
| Norm position | ? | Pre-norm | |
| Activation | ? | SwiGLU | |
| Position embedding | ? | RoPE | |
| $d_{ff}/d_{model}$ | ? | 8/3 | |
| Bias terms | ? | None | |
| Attention type | ? | GQA | |

Then discuss: where does this model deviate from the consensus, and why?

### Exercise 2: Parameter Budget Calculator (30 min)
Given a fixed parameter budget (e.g., 3B, 7B, 13B), have students derive the full architecture specification using the consensus rules. Then compare their answer to real models of the same size.

```python
# Constraints:
# d_ff = 8/3 * d_model (SwiGLU)
# d_head = 128
# n_heads = d_model / d_head
# d_model / n_layers ≈ 128
#
# Total params ≈ n_layers * (
#   3 * d_model * d_ff        # SwiGLU MLP: W, V, W2
#   + 4 * d_model * d_model   # Attention: Q, K, V, O projections
#   + 2 * d_model              # RMSNorm (before attention + before MLP)
# ) + vocab_size * d_model     # Embedding
```

### Exercise 3: Stability Detective (Discussion, 15 min)
Show students a training loss curve with gradient norm spikes (like the OLMo blue curve). Ask them to:
1. Identify the likely cause (softmax instability, activation growth, etc.)
2. Propose 3 interventions from the lecture (z-loss, QK-norm, logit capping)
3. Predict which would help most and why
4. Discuss what would happen if they just retrained from the last checkpoint before the spike

### Exercise 4: RoPE Implementation (Homework)
Have students implement RoPE from scratch in PyTorch (no library calls):
1. Generate the rotation matrix for given positions and dimensions
2. Apply it to query and key tensors
3. Verify the relative position property: show that $\langle R_m q, R_n k \rangle = \langle R_0 q, R_{n-m} k \rangle$
4. Bonus: Implement NTK-aware scaling for context length extension
