# Implementing a LLM

 - In previous classes we have covered tokenization, embeddings, and the transformer architecture in detail.
- We have also implemented the multi head self-attention mechanism, the crucial component of transformers.

- In this class we will put everything together and implement a small LLM from scratch using PyTorch.

- The implmentation closely follows the GPT-2 architecture.


## GPT - 2 Model
GPT-2 model is built using the transformer decoder blocks.
The model is just a stack of transformer decoder blocks, with an embedding layer at the input and a linear + softmax layer at the output.

How many layers of transformer blocks?
 - GPT-2 Small: 12 layers
 - GPT-2 Medium: 24 layers
 - GPT-2 Large: 36 layers
 - GPT-2 XL: 48 layers

Generation Viz 1:

 ![](https://jalammar.github.io/images/xlnet/gpt-2-output.gif)


Generation Viz 2:
![](https://jalammar.github.io/images/xlnet/gpt-2-autoregression-2.gif)


### Looking inside the GPT-2 model
- A stack of transformer blocks
- Each block has multi-head self-attention, feed forward network, layer normalization, and residual connections
- Input embeddings + positional encodings at the bottom, linear + softmax layer at the top

![alt text](https://jalammar.github.io/images/gpt2/gpt-2-layers-2.png)

Sample Text generation

![](https://jalammar.github.io/images/gpt2/gpt-2-simple-output-3.gif)
- Initally the model only has one input token, so that path would be the only active one. 
- The token is processed successively through all the layers, then a vector is produced along that path. That vector can be scored against the model’s vocabulary (all the words the model knows, 50,000 words in the case of GPT-2) and the most likely next token can be selected.
- next step, we add the output from the first step to our input sequence, and have the model make its next prediction:
- Each layer of GPT-2 has retained its own interpretation of the first token and will use it in processing the second token (we’ll get into more detail about this in the following section about self-attention). GPT-2 does not re-interpret the first token in light of the second token.

**Input Encoding**
- We need to give the model input in the form of token IDs.
- for each token ID, we look up its corresponding embedding vector from the embedding matrix.

![](https://jalammar.github.io/images/gpt2/gpt2-token-embeddings-wte-2.png)

- We take mbeddings and add positional encodings to them to give the model information about the position of each token in the sequence.

![positional encoding](https://jalammar.github.io/images/gpt2/gpt2-positional-encoding.png)

After combination: 
![Input + position encoding](https://jalammar.github.io/images/gpt2/gpt2-input-embedding-positional-encoding-3.png)


In summary, the input to GPT-2 is a sequence of token IDs, which are converted to embedding vectors and combined with positional encodings to form the final input representation for the model.

**Passing token through the transformer blocks**
- The first block can now process the token by first passing it through the self-attention process, then passing it through its neural network layer. 
- Once the first transformer block processes the token, it sends its resulting vector up the stack to be processed by the next block. 
- The process is identical in each block, but each block has its own weights in both self-attention and the neural network sublayers.
![alt text](https://jalammar.github.io/images/gpt2/gpt2-transformer-block-vectors-2.png)

**Self-Attention in GPT-2**
- Self attention allows each token to pay importance to other tokens in the sequence when processing itself.
- Example 1: ”The animal didn't cross the street because it was too tired” . When the model is processing the word “it”, self-attention allows it to associate “it” with “animal”.
- Example 2 : "A robot must obey the orders given **it** by human beings except where **such orders** would conflict with **the First Law.**” Here, self-attention helps the model understand that "such orders" refers to "the orders given it by human beings".

![](https://jalammar.github.io/images/gpt2/gpt2-self-attention-example-2.png)

**Self-Attention Process**
- For each token, we create Query (Q), Key (K), and Value (V) vectors
- Query: The query is a representation of the current word used to score against all the other words (using their keys). We only care about the query of the token we’re currently processing.
- Key: Key vectors are like labels for all the words in the segment. They’re what we match against in our search for relevant words.
- Value: Value vectors are actual word representations, once we’ve scored how relevant each word is, these are the values we add up to represent the current word.

![](https://jalammar.github.io/images/gpt2/self-attention-example-folders-scores-3.png)
- We compute attention scores by taking the dot product of the Query vector of the current token with the Key vectors of all tokens.
- We scale the scores by dividing by the square root of the dimension of the Key vectors.
- We apply softmax to the scores to get attention weights.
- We multiply each Value vector by its corresponding attention weight.
- We sum the weighted Value vectors to get the output vector for the current token.

![](https://jalammar.github.io/images/gpt2/gpt2-value-vector-sum.png)


**Model Output**
 - After the input token has passed through all the transformer blocks, we get a final output vector. 
 ![](https://jalammar.github.io/images/gpt2/gpt2-output-projection-2.png)

- This vector is then passed through a linear layer followed by a softmax layer to produce the final output probabilities for each token in the vocabulary.

![](https://jalammar.github.io/images/gpt2/gpt2-output.png)


![alt text](01_llm_arch.png)

Our imlementations:
vocabulary size: 50257 (same as GPT-2)
embedding dimension: 768
number of transformer blocks: 12
number of attention heads: 12
maximum sequence length: 1024


## Transformer Block

## Layer Normalization

## Feed Forward Network (FFN)

## Residual Connections / shortcuts
    
## Illustrated GPT-2 Architecture

## Generating Text 
as we can see, the transformer Block maintains the input dimensions in its output, indicating that the transformer architecturfe proc esses sequences of data without altering their shape through out the network

## Pretraining the LLM


### Tidbits
 - show cross entropy loss calculation for next token prediction
    - on random token predction on the vocabulary
    - show as model trains 
    - show on trained weights

- 