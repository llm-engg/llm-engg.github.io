<div class="center-slide">

## LLMs : A Hands-on Approach 

### Transformers
</div>

---

## Topics Covered

- Transformer architecture
- Self-attention mechanism
- Causal Attention
- Multi-head attention

---

## Models of the Week

**[moonshotai/Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5)**

- SOTA Open Source model
- 1T+ parameters, 30T tokens pretraining data, Vocabulary Size - 160K, Context Length - 256K
- Agentic, Agent Swarm, Multi-modal capabilities


**[nvidia/personaplex-7b-v1](https://research.nvidia.com/labs/adlr/personaplex/)**

- 7B, Real-time, speech to speech model, full-duplex model 
- [Demo](https://research.nvidia.com/labs/adlr/personaplex/)


**[Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)**

- Text to Speech model with voice design, voice cloning, custom voice

---

## Recap : Tokenization

 - Words, Subwords, Characters level tokenization
 - Subword tokenization is the most commonly used approach in LLMs

![](images/tokens.png)

---

### Text to Token IDs


```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
text = "This is an example."
token_ids = tokenizer.encode(text)
# tokens = ['This', ' is', ' an', ' example', '.']
# token_ids = [40234, 2052, 133, 389, 12]
```

![](images/word_tokenization.png)

---

### Tokens to Embeddings
- Each token ID maps to a unique vector in the embedding matrix
- Embedding Matrix : [vocab_size x embedding_dim] 
- Example : GPT-2 Small
    - vocab_size = 50257
    - embedding_dim = 768

<div style="display: flex; justify-content: space-between; align-items: center;">
    <div style="flex: 1; padding: 10px;">
        <img src="images/lm_token_embeddings.png" style="max-width: 50%;">
    </div>
    <div style="flex: 1; padding: 10px;">
        <img src="images/emb-matrix.png" style="max-width: 50%;">
    </div>
</div>

---

<div class="center-slide">

## Transformers

</div>

---
###  Transformers

<img src="images/transformer.png" class="float-right">

- All LLMs rely on the Transformer architecture, introduced in the 2017 paper "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).
- No Recurrent or Convolution layers, entirely based on Attention Mechanism

---

### Two types of Transformer Architectures

<img src="images/enc-dec-vs-dec-only.png" class="float-right">


- **Encoder-Decoder** : Sequence to sequence tasks (translation, summarization)
- **Decoder-Only** : Language modeling tasks (text generation, completion)
- Most LLMs (GPT, Llama, etc.) use the **Decoder-Only** architecture

---
### Key Components of Transformer

<img src="images/enc-dec.png" class="float-right">

- Positional Encoding <!-- .element: class="fragment semi-fade-out" data-fragment-index="1" -->
- Multi-Head Attention <!-- .element: class="fragment highlight-current-blue grow" data-fragment-index="1" -->
- Residual Connections <!-- .element: class="fragment semi-fade-out" data-fragment-index="1" -->
- Feed Forward Networks <!-- .element: class="fragment semi-fade-out" data-fragment-index="1" -->
- Layer Normalization <!-- .element: class="fragment semi-fade-out" data-fragment-index="1" -->

---

### Language Modeling

- Given a sequence of tokens, predict the next token
- Example: A robot may not harm a ___ -> human
- [PROMPT] -> [MODEL] -> [PREDICTION]

![](https://jalammar.github.io/images/xlnet/gpt-2-autoregression-2.gif)

---

### Decoder-only Language Model

- A decoder-only language model is a stack of transformer decoder blocks

![alt text](https://jalammar.github.io/images/gpt2/gpt-2-simple-output-3.gif)

---

### Decoder-only LLM : Input Side

- Input tokens are passed through multiple  decoder blocks
- **Embed** : Text -> Token IDs -> Embeddings -> Decoder Blocks

![alt text](images/decoder-blocks.png)

---

### Decoder-only LLM : Output Side

- **UnEmbed** : The final vector is projected to vocabulary size and softmaxed to get token probabilities

![](images/output-layer.png)

---

### Decoder Block Internals

- Masked Multi-Head Self-Attention
- Feed Forward Network (FFN)
- Residual Connections
- Layer Normalization

![alt text](images/decoder-internals.png)


---

<div class="center-slide">

## Attention Mechanism

</div>

---

### Sequence Modeling Challenges

- Understanding context and relationships between words
- Need to keep grammatical structures aligned

![](images/translate.png)

---

### Recurrent Neural Networks (RNNs) for Sequence Modeling

- Process all input into a hidden state, 
- Pass hidden state to decoder
- Decoder uses hidden state to generate output sequence

 ![Encoder-Decoder](images/rnn.png)

---

### RNNs + Attention

- Let Decoder access all Encoder hidden states
- Attend to relevant parts of input sequence when generating each output token [Bahdanau et al., 2015]

![bahdanau-attn](https://camo.githubusercontent.com/90bef5f34f4eb3eb23e8446eb150507bb0df2fc5cd4f0c5dc7e9d4db3dab1058/68747470733a2f2f332e62702e626c6f6773706f742e636f6d2f2d3350626a5f64767430566f2f562d71652d4e6c365035492f41414141414141414251632f7a305f365774565774764152744d6b3069395f41744c6579794779563641493477434c63422f73313630302f6e6d742d6d6f64656c2d666173742e676966)

---

### RNNs + Attention

**Limitations**

- Sequential processing limits parallelization
- Difficulty capturing long-range dependencies

**Solution**: 

- Remove recurrence, process all input tokens simultaneously
- Allow each token in the input to focus on relevant parts of the input

---

### Parallel Processing

- **Encoder** : Process all input tokens simultaneously
- **Decoder** : Generate output tokens one by one, attending to encoder states and previous tokens

![](https://3.bp.blogspot.com/-aZ3zvPiCoXM/WaiKQO7KRnI/AAAAAAAAB_8/7a1CYjp40nUg4lKpW7covGZJQAySxlg8QCLcBGAs/s640/transform20fps.gif)

---

### Self-Attention Mechanism

- Compute attention **within** the same sequence of tokens. **Self = Same Sequence**
- Get improved representation by **mixing in information** from other tokens **that seem relevant.**

![alt text](images/self-attn.png)

---


###  Self-Attention : Intuition


<div style="text-align: center;">

*"Self-attention is like a group conversation where everyone can hear
everyone else simultaneously, rather than passing notes one by one (RNNs)"*

</div>

    

---

###  Self-Attention : Intuition

<img src="images/attn-weighted.png" class="float-right">

- Each token : "Who should I pay attention to?"

- For every token, the model:
    - treats that token as the "current focus"
    - assigns **higher weight** to tokens that help interpret it
    - creates an updated vector for the token:  


---


### Self-Attention vs Encoder–Decoder Attention

- **Encoder–Decoder**: One sequence attends to a *different* sequence (e.g., translation: output attends to the input sentence).

- **Self-attention**: Sequence attends to **itself** (tokens attending to other tokens in the same sentence).


<div style="display: flex; justify-content: space-between; align-items: center;">
    <div style="flex: 1; padding: 10px;">
        <img src="images/bahdanau-attn.png" style="max-width: 50%;">
    </div>
    <div style="flex: 1; padding: 10px;">
        <img src="https://jalammar.github.io/images/t/transformer_self-attention_visualization.png" style="max-width: 50%;">
    </div>
</div>

---

### Simple Attention Mechanism

<img src="images/simple-attention.png" class="float-right">

**Input**  -  Sequence of vectors (X) (source)

**Output** - Sequence of vectors (Z) (context)

<div style="text-align: center;">
$$
X = [x_1, x_2, \dots, x_n], \quad x_i \in \mathbb{R}^d
$$

$$
Z = [z_1, z_2, \dots, z_n], \quad z_i \in \mathbb{R}^d
$$

$$
z_i = \sum_{j=1}^{n} \text{attention\_weight}_{ij} \. x_j
$$
</div>

---

### Computing attention weights for a single token

<div style="text-align: center;">

*Your **journey** starts with one step*

</div>

query = "journey"

<img src="images/simple-attn-scores.png" class="float-right">

Step 1: 

- Compute attention scores by dot product of "journey" with all tokens

```python 
query = inputs[1]

attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

print(attn_scores_2)
```

--

### Computing attention weights for a single token

<img src="images/normalized.png" class="float-right">

Step 2:

- Apply normalization to get attention weights (additive normalization)
- Normalization using softmax is more common in practice, as it ensures all weights are positive and sum to 1.


```python
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

```

--

Step 3:
- Compute output vector as weighted sum of value vectors

![alt text](images/context-vector.png)

```python
context_vec_2 = torch.zeros(inputs.shape[1])
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
```

---

### Computing attention weigths for all tokens

- Compute attention scores for all tokens
```python [1 | 2-10]
attn_scores = torch.zeros(inputs.shape[0], inputs.shape[0])

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

```

- Normalize scores to get attention weights
```python
attn_weights = torch.softmax(attn_scores, dim=-1)
```

- Compute output/context    vectors for all tokens
```python
output_vectors = torch.zeros_like(inputs)
for i in range(inputs.shape[0]):
    for j in range(inputs.shape[0]):
        output_vectors[i] += attn_weights[i, j] * inputs[j]
```

--

### Computing attention weigths for all tokens

- Better implementation using matrix multiplication

```python

attn_scores = torch.zeros(inputs.shape[0], inputs.shape[0])

attn_scores = inputs @ inputs.T
attn_weights = torch.softmax(attn_scores, dim=-1)
output_vectors = attn_weights @ inputs

```

---

### Summary of Self-Attention Mechanism
- Input: sequence of vectors (X) (source)
- Output: sequence of vectors (Z) (context)
- Compute attention scores against all input vectors
- Normalize scores to get attention weights
- Compute output vectors as weighted sum of input vectors

```python
def self_attention(inputs):
    # Step 1: Compute attention scores
    attn_scores = inputs @ inputs.T
    # Step 2: Normalize scores to get attention weights
    attn_weights = torch.softmax(attn_scores, dim=-1)
    # Step 3: Compute output vectors as weighted sum of input vectors
    output_vectors = attn_weights @ inputs
    return output_vectors

```

<div style="text-align: center;">

**How to improve this basic self-attention mechanism?** <!-- .element: class="fragment" data-fragment-index="2" -->

Learn the weights used to compute attention scores! <!-- .element: class="fragment" data-fragment-index="3" -->

</div>

---

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) - The original transformer paper
- Bahdanau et al., [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (2014) - Introduced attention for seq2seq
- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Jay Alammar, [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Harvard NLP
- Sebastian Raschka, *Build a Large Language Model from Scratch* - Chapters 3-4


---

## Thank You

Questions?
