# Mixture of Experts (MoE) in LLMs

Notes from Stanford CS336 Lecture 4 (Spring 2025) by Tatsunori Hashimoto.
Sources: [Lecture video](https://www.youtube.com/watch?v=LPv1KfUXLCo), [Lecture slides](https://github.com/stanford-cs336/spring2025-lectures)

---

## Why MoEs Matter Now

Nearly every frontier LLM is an MoE: GPT-4 (rumored via the NVIDIA leak of "GPT-MoE-1.8T"), Grok, Mixtral, DeepSeek V2/V3, Llama 4, DBRX, Qwen, and OLMoE. MoEs went from a niche research topic to the dominant architecture for high-performance open models in under two years. As Hashimoto puts it: "at almost all compute scales, training a mixture of experts model, if you do it well, is going to give you benefits over a dense model."

Three reasons MoEs are winning:

1. **Same FLOPs, more parameters, better performance.** At fixed compute, adding more experts consistently lowers loss. Switch Transformer (Fedus et al. 2022) showed that going from 1 expert to 256 experts monotonically reduces test loss at the same FLOP budget. "As you increase the number of experts, the training loss just keeps going down and down." If you believe what matters is having more parameters to memorize facts about the world, MoE is a great architecture.

2. **Faster to train.** OLMoE showed a 1.3B-active / 6.9B-total MoE achieves ~3x less FLOPs or tokens to match a dense 1.3B model on HellaSwag, and trains ~2x faster wall-clock. Switch Transformer 128-expert model reached the same perplexity as T5-Base with a **7x training speedup**.

3. **Naturally parallelizable.** Each expert can live on a separate device. Expert parallelism is a new dimension beyond data/model/pipeline parallelism -- experts on different GPUs process their assigned tokens independently, connected by all-to-all communication. Because experts are sparsely activated, all you do is route each token to the appropriate device and the computation happens there.

**Benchmark evidence:** DeepSeek-V2 matches LLaMA 3 70B on MMLU with only ~21B activated parameters. Llama 4 Maverick beats Gemini 2.0 Flash and GPT-4o on multiple benchmarks at a fraction of the inference cost ($0.19-$0.49 per 1M tokens vs $4.39 for GPT-4o).

### Why Haven't MoEs Been More Popular Earlier?

- **Infrastructure complexity** -- advantages mainly materialize at multi-node scale with many accelerators. When you have to split up your models anyway, it makes sense to shard experts across different devices. But until you get to that point, MoEs aren't clearly better. "They're very complex and very messy."
- **Training instability** -- sparse models suffer from worse training instabilities than dense transformers (loss spikes, router collapse). The routing decision (which expert to send a token to) is a discrete, non-differentiable choice -- fundamentally at odds with the smooth gradients deep learning relies on.
- **Fine-tuning challenges** -- MoEs overfit more easily on small fine-tuning datasets due to the large total parameter count relative to activated parameters.

### A Misleading Name

"Mixture of Experts" is a terribly named concept. You hear it and think there must be experts specialized for different domains -- a coding expert, an English expert, etc. It is very far from that mental model. MoE is simply a sparse architecture with multiple sub-networks activated selectively. The "experts" don't learn clean semantic specializations.

---

## Core Architecture

### Dense vs Sparse Transformer

In a **dense** transformer, every token passes through one large FFN block. In a **sparse** (MoE) transformer:

- The single FFN is replaced by $N$ smaller "expert" FFN sub-networks
- A **router** (gating network) selects only $K$ of them per token
- The outputs of the selected experts are combined via weighted sum

**Key insight:** You can increase the number of experts without affecting FLOPs. Only $K$ experts are activated per token, so compute stays roughly constant while total parameters grow.

### What Gets Replaced

- **Typical:** Replace the MLP/FFN layer with MoE layer (most models do this)
- **Less common:** MoE for attention heads (ModuleFormer, JetMoE) -- each attention head is also routed

The attention layers remain shared/dense in most architectures. The MoE action is specifically in the MLPs.

---

## MoE Design Dimensions

Three things vary across MoE architectures:

1. **Routing function** -- how tokens get assigned to experts
2. **Expert sizes** -- how many experts, how large each one is, shared vs routed
3. **Training objectives** -- how to train the router and keep experts balanced

---

## Routing Mechanisms

### Three Routing Paradigms

| Paradigm | How it works | Who uses it |
|---|---|---|
| **Token chooses expert** | Each token picks its top-K experts | Most models (standard) |
| **Expert chooses token** | Each expert picks its top-K tokens | Naturally balanced, but variable tokens/expert |
| **Global routing** | Solve an optimization/matching problem over all tokens and experts | Theoretical, rarely used |

Almost all production MoEs use **token-choice top-K routing**. Ablations from OLMoE show token choice (TC) and expert choice (EC) perform similarly, with TC having a slight edge on downstream tasks.

### Top-K Routing in Detail

The standard formulation (used in DeepSeek V1-V2, Grok, Qwen, Mixtral, DBRX):

$$\mathbf{h}_t^l = \sum_{i=1}^{N} \left( g_{i,t} \cdot \text{FFN}_i(\mathbf{u}_t^l) \right) + \mathbf{u}_t^l$$

where the gating weights are:

$$g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \text{TopK}(\{s_{j,t} | 1 \leq j \leq N\}, K) \\ 0, & \text{otherwise} \end{cases}$$

$$s_{i,t} = \text{Softmax}_i\left(\mathbf{u}_t^{lT} \mathbf{e}_i^l\right)$$

The router is just a **logistic regressor** -- a linear projection from hidden state to expert scores, followed by softmax. The simplicity is notable: complex routing (RL-based, learned routing networks) provides only marginal benefit. The router operates on the hidden state (after position embeddings, attention, etc.), not on raw token IDs.

**Why is the router so basic?** (Q&A) Two reasons: (1) Systems concerns -- more FLOPs spent on routing means less available for actual computation, and (2) there are hard limits on how well you *can* route, because the learning signal for routing is very indirect. With top-2, you can only compare the two experts you actually evaluated. Even making the router an MLP doesn't clearly help.

**Why not just softmax without top-K?** You immediately lose the systems efficiency. Without top-K, you pay the training cost of all N experts per token -- the whole point of MoE is sparse activation during both training and inference.

**Why softmax before top-K?** The softmax here is really a "normalize to one" operation to make the gating weights a proper weighted average. After top-K, the weights no longer sum to one -- some architectures renormalize after top-K, some don't. It doesn't matter much since subsequent layer norms can adjust scale.

**Why K >= 2?** The original argument was exploration: with K=1, you always exploit the best expert and never learn about alternatives. K=2 gives you a second "arm" that provides exploration signal, like epsilon-greedy in bandits. K=2 doubles activated FLOPs, which is why "activated parameters" is the metric MoE papers report.

**How are K experts combined?** The outputs of the K selected experts are summed (weighted by gating scores). This is NOT an expectation over FFNs -- each FFN_i is a different function, and the gates are sparse.

### Common Top-K Values

| Model | K |
|---|---|
| Switch Transformer | 1 |
| GShard, Grok, Mixtral | 2 |
| Qwen, DBRX | 4 |
| DeepSeek V1 | 6 |
| DeepSeek V3 | 8 (of 256) |
| OLMoE | 8 (of 64) |

### Hash Routing (Baseline)

A non-learned baseline where tokens are assigned to experts via a hash function. Surprisingly competitive -- "even if you're hashing, you will still get gains, which is pretty wild." Why does this work? Even with hashing, the same tokens consistently go to the same expert, so specialization still occurs (just non-semantic). For Zipfian distributions, frequent words like "the" might dominate one expert, giving accidental semantic clustering. A truly random routing (different expert each time, not input-dependent) would likely be terrible.

### Other Routing Methods

- **Reinforcement learning** (Bengio 2013): Use REINFORCE to learn routing policy. "It's probably the most principled thing you can do -- you have a non-differentiable routing decision, think of it as a policy, throw RL at it." But gradient variance and complexity make it impractical. Not clearly better than hashing. Basically abandoned.
- **BASE routing** (Clark 2022): Solve a linear assignment problem for globally optimal token-to-expert matching. Elegant but too expensive for the marginal benefit.

---

## Expert Configuration

### Fine-Grained Experts (DeepSeek Innovation)

Instead of $N$ full-sized FFN copies, make each expert **much smaller** (1/4 to 1/14 of standard FFN size) and use **many more** of them. This enables finer-grained specialization.

The logic: "lots of experts is good" -> "I want lots of experts but don't want to pay the parameter cost" -> cut each expert into smaller pieces. If the standard FFN has a 1:4 hidden-to-intermediate ratio, you can use 1:2 instead (half the size), doubling the expert count for the same total parameters. Since each fine-grained expert is smaller, having more active experts is FLOPs-free.

The fine-grained ratio = (expert intermediate dim) / (standard FFN intermediate dim).

**Example:** DeepSeek V1 has 64 experts at 1/4 ratio with 6 routed + 2 shared = 8 active. Each active expert is quarter-sized, so total active computation is roughly 2x a dense FFN.

### Shared Experts

One or more experts that process **all tokens** regardless of routing. The idea: maybe some processing always needs to happen no matter which token you're seeing. Having a shared expert dedicated to this avoids wasting routing decisions on universal computation.

**Mixed evidence:** DeepSeek ablations show shared experts help. OLMoE ablations show no benefit. There was a period where Chinese LM companies tried many shared experts (Qwen had 4), but the field has converged to 0 or 1. The original motivation for 2 shared experts (DeepSeek V1) was to keep all experts the same size -- two quarter-sized shared experts instead of one half-sized one.

### Expert Routing Configurations for Major MoEs

| Model | Routed | Active | Shared | Fine-grained ratio |
|---|---|---|---|---|
| GShard | 2048 | 2 | 0 | -- |
| Switch Transformer | 64 | 1 | 0 | -- |
| Mixtral | 8 | 2 | 0 | -- |
| DBRX | 16 | 4 | 0 | -- |
| Grok | 8 | 2 | 0 | -- |
| DeepSeek V1 | 64 | 6 | 2 | 1/4 |
| Qwen 1.5 | 60 | 4 | 4 | 1/8 |
| DeepSeek V3 | 256 | 8 | 1 | 1/14 |
| OLMoE | 64 | 8 | 0 | 1/8 |
| MiniMax | 32 | 2 | 0 | ~1/4 |
| Llama 4 (Maverick) | 128 | 1 | 1 | 1/2 |

### Ablation Results

**DeepSeek ablations:** Fine-grained expert segmentation AND shared expert isolation both contribute to stronger performance across HellaSwag, PIQA, ARC, TriviaQA, NaturalQuestions. Going from 0 shared + 2/16 routed (GShard-style) to 1 shared + 7/63 routed (finer segmentation) progressively improves all benchmarks.

**OLMoE ablations:** Gains from fine-grained experts confirmed. However, shared experts showed **no benefit** in their setup (32 routed vs 31 routed + 1 shared performed the same). Increasing expert count from 8 to 32 to 64 consistently improved results.

---

## Training MoEs

### The Core Challenge

We need sparsity for training-time efficiency, but **sparse gating decisions are not differentiable** (hard top-K selection). If we turn on all experts, we pay the full FLOPs cost -- "having a model that's 256 times more expensive to train is a total no-go."

Three approaches:
1. **Reinforcement learning** to optimize gating policies -- the "right" solution theoretically, but gradient variance and complexity make it impractical
2. **Stochastic perturbations** -- add noise to make routing soft/differentiable (bandit-style exploration)
3. **Heuristic balancing losses** -- what everyone actually uses in practice

"Having gone through deep learning classes of many kinds, you can kind of guess internally which one people use in practice."

### Stochastic Perturbations

**Shazeer et al. (2017) -- Noisy top-K gating:**

$$G(x) = \text{Softmax}(\text{KeepTopK}(H(x), k))$$
$$H(x)_i = (x \cdot W_g)_i + \text{StandardNormal}() \cdot \text{Softplus}((x \cdot W_{\text{noise}})_i)$$

where $\text{KeepTopK}$ sets non-top-K values to $-\infty$. The Gaussian noise encourages exploration of different expert assignments.

**Fedus et al. (2022) -- Multiplicative jitter:**

```python
if is_training:
    router_logits += mtf.random_uniform(shape=router_logits.shape,
                                         minval=1-eps, maxval=1+eps)
router_logits = mtf.to_float32(router_logits)  # float32 for stability
router_probs = mtf.softmax(router_logits, axis=-1)
```

This was later found to hurt quality slightly and removed in Zoph et al. (2022). Input jitter improves stability (3/3 stable vs 4/6 baseline) but at a quality cost (-1.777 vs -1.755).

### Heuristic Balancing Losses

Without load balancing, routers collapse to using only 1-2 experts, wasting all other capacity. The OLMoE paper shows this clearly: without load balancing loss (LBL), "the model just picks one or two experts and all the other experts are dead. The router never sends anything to them. So you're just wasting memory -- you've effectively gotten a smaller model." Even ignoring systems concerns, you want expert balancing just to use all your parameters effectively.

#### Switch Transformer F*P Loss (Standard)

$$\mathcal{L}_{\text{balance}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

where:
- $f_i = \frac{1}{T} \sum_{x \in \mathcal{B}} \mathbb{1}\{\text{argmax } p(x) = i\}$ is the fraction of tokens dispatched to expert $i$
- $P_i = \frac{1}{T} \sum_{x \in \mathcal{B}} p_i(x)$ is the mean router probability for expert $i$
- $\alpha$ is the balancing coefficient
- $N$ is the number of experts

**Why it works:** The derivative w.r.t. $p_i(x)$ is $\frac{\alpha N}{T^2} \sum \mathbb{1}_{\text{argmax } p(x)=i}$, so **more frequent use = stronger downweighting**. This creates a natural pressure toward uniform distribution.

**Tradeoff:** Large $\alpha$ improves balance but introduces interference gradients that harm model quality.

#### DeepSeek V1-V2: Multi-Level Balancing

Two levels of balancing:

**Per-expert balancing** (same as Switch Transformer):

$$\mathcal{L}_{\text{ExpBal}} = \alpha_1 \sum_{i=1}^{N'} f_i P_i$$

**Per-device balancing** (reduces communication costs):

$$\mathcal{L}_{\text{DevBal}} = \alpha_2 \sum_{i=1}^{D} f_i' P_i'$$

where $f_i'$ and $P_i'$ aggregate over experts on device $i$.

#### DeepSeek V3: Auxiliary-Loss-Free Balancing

A novel approach that avoids the quality-balance tradeoff:

1. Add a **learned bias** $b_i$ to each expert's routing score
2. Use **online gradient descent per batch** (not backprop) to update $b_i$
3. If expert $i$ gets too many tokens, decrease $b_i$; if too few, increase it

$$g'_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} + b_i \in \text{TopK}(\{s_{j,t} + b_j | 1 \leq j \leq N_r\}, K_r) \\ 0, & \text{otherwise} \end{cases}$$

A **complementary sequence-wise auxiliary loss** is still added for stability (so it's not fully auxiliary-loss-free). "If you read the DeepSeek V3 paper, they make a big deal about how this makes training so stable, so great. And then you keep reading and they're like, actually we decided we needed the heuristic loss back."

$$\mathcal{L}_{\text{Bal}} = \alpha \sum_{i=1}^{N_r} f_i P_i$$

where $f_i$ and $P_i$ are computed per-sequence rather than per-batch.

**Why per-sequence?** At inference time, you can't control which sequences you receive. Out-of-distribution inputs might overwhelm certain experts. Sequence-level balancing provides stronger guarantees than batch-level when individual sequences are adversarial or unusual.

---

## Stability Issues

### Router Softmax Overflow

Exponential functions amplify small perturbations. In bfloat16, a roundoff error of 0.5 on logit values around 128 can alter the softmax output by 36%.

**Solution:** Always compute routing in **float32**, even when the rest of the model uses bf16/fp16.

### Z-Loss Regularization

A penalty on the log-sum-exp normalizer of router logits that prevents logit values from growing too large:

$$L_z(x) = \frac{1}{B} \sum_{i=1}^{B} \left( \log \sum_{j=1}^{N} e^{x_j^{(i)}} \right)^2$$

OLMoE ablations show that without z-loss, training suffers severe instability -- massive spikes in validation loss and HellaSwag performance that z-loss completely eliminates. Z-loss weight of 0.001 works well.

### Token Dropping and Stochasticity

When an expert receives more tokens than its **capacity factor** allows, excess tokens are dropped. This creates a source of randomness even at temperature 0 during inference -- other queries in the same batch can affect which of your tokens get dropped.

This was speculated to be the source of GPT-4's observed stochasticity (when temperature=0 still gave different outputs). Token dropping happens at the **batch level**, meaning other people's queries can cause your tokens to be dropped -- a cross-batch effect you almost never think about in standard inference. If your batch happens to have many tokens that love expert 3, and the device for expert 3 doesn't have enough memory, some tokens get dropped and receive zero MLP computation (just the residual connection passes through).

---

## Systems: Training MoEs at Scale

### Expert Parallelism

MoEs enable a new dimension of parallelism beyond data/model/pipeline:

1. **Compute routing decisions** locally on each device
2. **All-to-all dispatch:** send tokens to the device hosting their assigned expert
3. **Parallel FFN computation:** each device runs its local experts on received tokens
4. **All-to-all gather:** collect outputs and route back to originating device

This combines with existing parallelism strategies:
- **Data Parallelism:** replicate model, split data
- **Model Parallelism:** split model weights across devices
- **Expert + Data Parallelism:** different experts on different devices, data split across replicas
- **Expert + Model + Data Parallelism:** all three combined for largest-scale training

### Sparse Matrix Multiplication

MoE routing creates variable-size batches per expert. Three approaches:

1. **Batched matrix multiplication** -- pad all expert batches to same size, run in parallel. Wastes FLOPs on padding.
2. **Block diagonal matrix multiplication** -- frame expert computation as block-diagonal matmul. More efficient but still assumes equal-sized blocks.
3. **Block sparse matrix multiplication** -- express as sparse matmul that handles variable expert loads without padding. Used by **MegaBlocks** library (used in many open MoEs).

### DeepSeek V2 Top-M Device Routing

To reduce all-to-all communication, DeepSeek V2 adds a constraint: first select the **top-M devices** (GPU nodes with highest affinity scores), then select top-K experts only within those devices. This keeps most communication local.

---

## Issues with Fine-Tuning MoEs

Sparse MoEs **overfit** more easily on smaller fine-tuning datasets. The total parameter count is much larger than what's activated per token, creating a capacity-data mismatch.

**Solutions:**
- **Freeze MoE experts, fine-tune non-MoE layers** (Zoph et al.): Only update attention and non-MoE FFN layers. SuperGLUE scores stay competitive (86 vs 86.2) while avoiding overfitting.
- **Use more fine-tuning data** (DeepSeek approach): DeepSeek uses 1.4M SFT examples covering math, code, writing, QA, reasoning, summarization to avoid overfitting their MoE.

---

## Upcycling: Dense-to-MoE Conversion

A cost-effective alternative to training MoE from scratch:

1. Start from a **pretrained dense model**
2. Copy the FFN weights to initialize $N$ experts
3. Apply small **perturbations** to each copy (break symmetry)
4. Initialize the **router from scratch**
5. Continue pretraining

### Upcycling Results

**MiniCPM-MoE (13.6B total):** Upcycled from MiniCPM-2.4B with top-K=2, 8 experts, ~4B active params. Trained with ~520B additional tokens. Outperforms DeepSeekMoE 16B and Mistral-7B on most benchmarks (MMLU 58.80, GSM8K 61.56, HumanEval 51.05).

**Qwen MoE (14.3B total, 2.7B active):** Initialized from Qwen 1.8B with top-K=4, 60 experts, 4 shared. One of the first confirmed upcycling successes. Achieves MMLU 62.5, competitive with Mistral-7B (64.1) while using only 2.7B active parameters.

The upcycling approach is particularly attractive because it reuses expensive pretraining investment and converges faster than training from scratch.

---

## DeepSeek Architecture Evolution

The lecture traces the full evolution, noting that "DeepSeek V3 is not very different architecturally from the earliest DeepSeek models -- they had nailed the architecture when training much smaller 2B parameter models."

| | DeepSeek V1 | DeepSeek V2 | DeepSeek V3 |
|---|---|---|---|
| Total params | 16B | 236B | 671B |
| Active params | 2.8B | 21B | 37B |
| Experts | 64 fine-grained | -- | 256 |
| Routed/token | 6 | -- | 8 |
| Shared experts | 2 | -- | 1 |
| Fine-grained ratio | 1/4 | -- | 1/14 |
| Gating | Softmax | Softmax | **Sigmoid** |
| Balancing | F*P aux loss | F*P + device balance | **Bias-based (aux-loss-free)** |
| Attention | Standard MHA | **Multi-Head Latent Attention** | MLA |
| Special | -- | Top-M device routing | Multi-token prediction |

**DeepSeek V3 MoE innovations:**
- **Sigmoid gating** instead of softmax (softer, doesn't force competition between experts). The gate is still normalized to sum to 1, just via a different mechanism.
- **Auxiliary-loss-free balancing** with learned per-expert bias offsets (but sequence-wise aux loss added back)
- Retains **top-M device routing** from V2, drops the communication balancing loss

**Key insight:** "DeepSeek V3 is not very different architecturally from the earliest DeepSeek models -- they had nailed the architecture when training much smaller 2B parameter models. They really just got the engineering right."

### Non-MoE Components of DeepSeek V3

**Multi-Head Latent Attention (MLA)** -- introduced in V2, an alternative to GQA for KV cache compression:

Instead of reducing the number of KV heads (GQA approach), MLA projects keys and values into a lower-dimensional **latent space C**:
- Input $h_t$ is projected to a small $C$ (compressed KV representation)
- Only $C$ is cached (much smaller than full K, V)
- K and V are reconstructed by up-projecting from $C$ when needed
- The clever trick: the up-projection matrix $W_{UK}$ can be **merged** with the query projection matrix via matrix associativity, so no extra FLOPs are needed

**Complication with RoPE:** RoPE rotation matrices sit between the query projection and the latent up-projection, breaking the matrix merge trick. DeepSeek's solution: apply RoPE only on non-compressed dimensions.

**Multi-Token Prediction (MTP)** -- a lightweight auxiliary task:
- A small one-layer transformer takes the hidden state and predicts one additional token ahead
- Despite the paper's elaborate diagram showing multi-token capability, "they only do MTP with one token ahead"
- Helps training signal but is a minor component

**DeepSeek V3 results:** Outperforms GPT-4o, Claude 3.5 Sonnet, and Llama 3.1 405B on MMLU-Pro (75.9), GPQA-Diamond (59.1), MATH 500 (90.2), AIME 2024 (39.2), Codeforces (51.6 percentile), and SWE-bench Verified (42.0).

---

## Key Takeaways

1. **MoE = replace FFN with N smaller expert FFNs + router.** Same FLOPs, more parameters, better performance. Don't think "specialized experts" -- think "sparse computation."
2. **Routing is embarrassingly simple.** A linear projection + softmax + top-K. Even hashing works. Complex routing (RL, optimal transport) provides marginal benefit at prohibitive cost.
3. **Load balancing is the real game.** Without it, routers collapse to 1-2 experts. The F*P auxiliary loss is standard; DeepSeek V3's bias-based approach is the latest (but they still add an aux loss back).
4. **Fine-grained experts are a no-brainer.** Smaller experts, more of them, same FLOPs. Shared experts are more debatable (DeepSeek says yes, OLMoE says no difference).
5. **Stability requires care:** float32 router computation + z-loss regularization. Without z-loss, expect severe training spikes.
6. **Upcycling works.** Copy a dense model's FFN, perturb, add a router, continue training. Cheap way to get MoE benefits.
7. **Expert parallelism** is a natural fit for distributed training. Top-M device routing (DeepSeek V2+) controls communication costs at scale.
8. **Architectures don't change much.** DeepSeek V1 to V3 is mostly the same MoE design, just scaled up with engineering improvements. "They nailed the architecture at the 2B scale."

---

## References

- Shazeer et al. (2017) -- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- Lepikhin et al. (2020) -- [GShard: Scaling Giant Models with Conditional Computation](https://arxiv.org/abs/2006.16668)
- Fedus et al. (2022) -- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
- Zoph et al. (2022) -- [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906)
- Clark et al. (2022) -- [Unified Scaling Laws for Routed Language Models](https://arxiv.org/abs/2202.01169)
- Dai et al. (2024) -- [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066)
- DeepSeek-AI (2024) -- [DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model](https://arxiv.org/abs/2405.04434)
- DeepSeek-AI (2024) -- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- Muennighoff et al. (2024) -- [OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060)
- Gale et al. (2023) -- [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)
