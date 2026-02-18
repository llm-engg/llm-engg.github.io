<div class="center-slide">

# LLMs : A Hands-on Approach 

### Modern Architectures
</div>

---

## Topics Covered

- **GPT-2 Review**
    - Training Loop
- **Modern LLM Architectures**
    - Norm Types
    - Activation Functions
    - Positional Encodings
    - Attention Variants
    - Hyperparameters

---


## Recap : GPT-2 Training Loop

During Training we update model weights to minimize loss through **backpropagation** and **gradient descent**.

![alt text](images/train-loop.png)


**Training Loop in code**

```python

    for epoch in range(num_epochs):
        model.train()  # Enable dropout

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset gradients

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()        # Calculate gradients
            optimizer.step()       # Update weights

```
---
##  Loading and Saving Model Weights


**We must save trained models to:**

- Avoid retraining
- Share models with others
- Resume training later
- Deploy to production


```python
# ============ SAVE ============

torch.save(model.state_dict(), "model.pth")

# Save model + optimizer (for resuming training)
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "model_and_optimizer.pth")


# ============ LOAD ============
# Load weights into fresh model
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()  # Set to evaluation mode

# Resume training
checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train()  # Set to training mode
```

---

## LLM Loss Surfaces

LLM training optimizes a **high-dimensional non-convex loss surface** defined by:


$$
\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \log P_\theta(x_i^{\text{target}})
$$

Key properties:

- Billions of parameters
- Extremely overparameterized
- Many equivalent minima
- Flat basins dominate

More details in :
 - [Unveiling the Basin-Like Loss Landscape in Large Language Models](https://arxiv.org/html/2505.17646v2)
 - [Visualizing the Loss Landscape of Neural Nets](https://www.youtube.com/watch?v=lyZorUc8Gm4)

---

<div class="center-slide">

## Modern Architectures

</div>

---

## GPT-2 Architecture

**Position embedding**: learned, absolute

**FFN**: GELU

$$ \text{FFN}(x) = GELU(xW_1 + b_1)W_2 + b_2 $$


**Norm type**: Pre-Norm, LayerNorm

![alt text](images/gpt-2.png)

---

## Current Models

<iframe src="https://docs.google.com/spreadsheets/d/e/2PACX-1vQsF7QOjxAI1f7ud_oYNLRBq6qa3ZzLqtMMF_1xOKKbi5qb6atwvgeYIp4pYjuGXHDTKXMO0IdxBaVw/pubhtml?gid=1330227995&amp;single=true&amp;widget=true&amp;headers=false" style="width:100%;height:100%;border:none;"></iframe>

---

## Llama 2, LlaMA 3 and Qwen 3 Architectures

**Position embedding**: RoPE (rotary position embeddings)

**FFN**: \*GLU variant (SwiGLU for LLaMA, GeGLU for Qwen)

$$
\textbf{SwiGLU}(x) = \text{Swish}(xW) \otimes (xV)W_2
$$

**Norm type**: Post-Norm, RMSNorm

![alt text](images/llama-2.png)

---

## Pre-Norm vs Post-Norm

**Almost all models post-2020 use pre-norm.**

![alt text](images/pre-post-ln.png)

**Original Transformer** : Post Norm

`x → Attention(x) → Add → LayerNorm → FFN → Add → LayerNorm`


**GPT 2** : Pre-Norm

`x → LayerNorm → Attention → Add → LayerNorm → FFN → Add`

---

## Pre-Norm vs Post-Norm

<image src="images/pre-post-ln.png" >

**Why pre-norm wins:**

- Better gradient flow throrugh residual connections. 
- Practical evidence: almost all modern LLMs use pre-norm


**Note** : Double norm also used in some models, but not as common as pre-norm. It applies LayerNorm both before and after the sub-layer.


*Question* : `BERT was trained with post-norm and it was huge success. But most models use pre-norm. Why?`

---

## LayerNorm vs RMSNorm


<div style="text-align: center;"> 
Strong consensus toward RMSNorm
</div>




<div style="display: flex; gap: 2rem;">

<div style="flex: 1;">

**LayerNorm** (original): 

Normalize by subtracting mean and dividing by std dev, then scale ($\gamma$) and shift ($\beta$):

$$y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} \cdot \gamma + \beta$$

*Models* : GPT-1/2/3, OPT, GPT-J, BLOOM

</div>

<div style="flex: 1; border-left: 2px solid #333; padding-left: 2rem;">

**RMSNorm** (modern): 

Drop the mean subtraction and bias term:

$$y = \frac{x}{\sqrt{||x||_2^2 + \epsilon}} \cdot \gamma$$

*Models* : LLaMA family, DeepSeek V3, Qwen3 etc

</div>


</div>

**Why RMSNorm**

 - Fewer operations: RMSNorm requires fewer computations (no mean subtraction, no bias term) which reduces both FLOPs and memory bandwidth.

---

## Dropping bias Terms in FFN and LayerNorm

Most modern transformers have **no bias terms** in linear layers or LayerNorm.

Original: $FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$

Modern: $FFN(x) = \sigma(xW_1)W_2$
SiLU activation is used instead of ReLU, but the key point is that **bias terms are removed**.

**Reasons:**

1. Same memory/data movement argument as RMSNorm -- fewer parameters to load
2. **Optimization stability** -- empirically, dropping bias terms stabilizes training of very large networks


***LayerNorm Recap***

- Most models use RMSNorm
- Almost all models use pre-norm

---

## Activations & Gated Linear Units (Strong trend toward SwiGLU/GeGLU)

**Evolution of activations:**

| Activation | Formula | Notable Models |
|---|---|---|
| ReLU | $FF(x) = \max(0, xW_1)W_2$ | Original transformer, T5, Gopher, OPT |
| GeLU | $FF(x) = GELU(xW_1)W_2$ where $GELU(x) = x\Phi(x)$ | GPT-1/2/3, GPT-J, BLOOM |
| SwiGLU | $FF(x) = (Swish(xW) \otimes xV)W_2$ | LLaMA 1/2/3, PaLM, Mistral, *most post-2023* |
| GeGLU | $FF(x) = (GELU(xW) \otimes xV)W_2$ | T5 v1.1, mT5, Phi3, Gemma 2/3 |


where `Swish(x) = x * sigmoid(x)` and $\otimes$ is elementwise multiplication.

---

## Gated Linear Units (GLU)

**What do GLUs do?**

- GLUs add a **gating mechanism**
- Hidden representation element-wise multiplied by a gate $xV$ (learned linear projection)
- $xV$ controls information flow through the MLP

$$\text{Standard:} \quad \sigma(xW_1) \rightarrow \sigma(xW_1) \otimes (xV) \quad \text{(gated)}$$

<div style="margin-bottom:30px"></div>

![alt text](images/glu.png)

---

## Gated Linear Units (GLU)


**More number of parameters?**

*The extra parameter V* means GLU models have 3 weight matrices *(W, V, W2)* instead of 2. 

How to keep parameter count the same? (memory is the real bottleneck, not compute)


---

## Gated Linear Units (GLU)


**More number of parameters?**

*The extra parameter V* means GLU models have 3 weight matrices *(W, V, W2)* instead of 2. 

How to keep parameter count the same? (memory is the real bottleneck, not compute)

**Scale the FF Params** 

---

## Gated Linear Units (GLU)


**More number of parameters?**

*The extra parameter V* means GLU models have 3 weight matrices *(W, V, W2)* instead of 2. 

How to keep parameter count the same? (memory is the real bottleneck, not compute)

**Scale the FF Params** 
- Standard MLP
    -  $W_1 \in \mathbb{R}^{d \times d_{ff}}$ + $W_2 \in \mathbb{R}^{d_{ff} \times d}$ = $2 \cdot d \cdot d_{ff}$ params
    - Total FFN params = $2 \cdot d \cdot 4 d$ = $8 \cdot d^2$.


---

## Gated Linear Units (GLU)


**More number of parameters?**

*The extra parameter V* means GLU models have 3 weight matrices *(W, V, W2)* instead of 2. 

How to keep parameter count the same? (memory is the real bottleneck, not compute)

**Scale the FF Params** 
- Standard MLP
    -  $W_1 \in \mathbb{R}^{d \times d_{ff}}$ + $W_2 \in \mathbb{R}^{d_{ff} \times d}$ = $2 \cdot d \cdot d_{ff}$ params
    - Total FFN params = $2 \cdot d \cdot 4 d$ = $8 \cdot d^2$.

- Gated MLP: 
    - $W \in \mathbb{R}^{d \times d_{ff}}$ + $V \in \mathbb{R}^{d \times d_{ff}}$ + $W_2 \in \mathbb{R}^{d_{ff} \times d}$ = $3 \cdot d \cdot d_{ff}$ params. 


---


## Gated Linear Units (GLU)


**More number of parameters?**

*The extra parameter V* means GLU models have 3 weight matrices *(W, V, W2)* instead of 2. 

How to keep parameter count the same? (memory is the real bottleneck, not compute)

**Scale the FF Params** 
- Standard MLP
    -  $W_1 \in \mathbb{R}^{d \times d_{ff}}$ + $W_2 \in \mathbb{R}^{d_{ff} \times d}$ = $2 \cdot d \cdot d_{ff}$ params
    - Total FFN params = $2 \cdot d \cdot 4 d$ = $8 \cdot d^2$.

- Gated MLP: 
    - $W \in \mathbb{R}^{d \times d_{ff}}$ + $V \in \mathbb{R}^{d \times d_{ff}}$ + $W_2 \in \mathbb{R}^{d_{ff} \times d}$ = $3 \cdot d \cdot d_{ff}$ params. 

- To match: 
    - set $d_{ff}^{gated} = \frac{2}{3} d_{ff}^{standard} = \frac{2}{3} \cdot 4d = \frac{8}{3}d$. 
    - Total FFN params = $3 \cdot d \cdot \frac{8}{3}d = 8 \cdot d^2$.

**Scaling Factors:** 
- Standard MLP: $d_{ff} = 4 \cdot d$
- Gated MLP: $d_{ff} = \frac{8}{3} \cdot d \approx 2.67 \cdot d$

---


## Serial vs Parallel Layers

**Normal transformer blocks are serial – they compute attention, then the MLP**


Standard transformer block can be written as:

$$ 
y = x + \text{MLP}(\text{LayerNorm}(x + \text{Attention}(\text{LayerNorm}(x))) 
$$

Whereas the parallel formulation can be written as:

$$ 
y = x + \text{MLP}(\text{LayerNorm}(x)) + \text{Attention}(\text{LayerNorm}(x)) 
$$

![alt text](images/parallel.png)

[image source](https://arxiv.org/html/2311.01906)

---


## Further Reading

[The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)