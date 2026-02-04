## Transformer Architecture : Introduction
Most modern LLMs rely on the transformer architecture, which is a deep neural network architecture introduced in the 2017 paper “Attention Is All You Need” (https://arxiv.org/abs/1706.03762).

TODO (cover transformer in depth)
Jay Alammar's blog : https://jalammar.github.io/illustrated-transformer/
Annotated transformer paper : http://nlp.seas.harvard.edu/2018/04/03/attention.html
Attention is all you need paper : https://arxiv.org/abs/1706.03762



### Encoder Architecture
**Self-Attention at high level**
 Say the following sentence is an input sentence we want to translate:

”The animal didn't cross the street because it was too tired”

What does “it” in this sentence refer to? Is it referring to the street or to the animal? It’s a simple question to a human, but not as simple to an algorithm.

When the model is processing the word “it”, self-attention allows it to associate “it” with “animal”.

**Self-Attention in detail**
- The first step in calculating self-attention is to create three vectors from each of the encoder’s input vectors (in this case, the embedding of each word). So for each word, we create a Query vector, a Key vector, and a Value vector. These vectors are created by multiplying the embedding by three matrices that we trained during the training process. 
    q1 = Wq x embedding1, k1 = Wk x embedding1, v1 = Wv x embedding1

- Next, we calculate a score that determines how much focus to place on other parts of the input sentence for each word. We do this by taking the dot product of the Query vector with the Key vector of each word in the sentence. This gives us a score for each word.
    score1 = q1 . k1, score2 = q1 . k2, score3 = q1 . k3, ...

- We then divide each of these scores by the square root of the dimension of the Key vectors
    score1 = score1 / sqrt(dk), score2 = score2 / sqrt(dk), ...
- Next, we apply a softmax function to these scores to obtain the weights on the Value vectors. The softmax function converts the scores into probabilities that sum to 1.
    weights = softmax([score1, score2, score3, ...])

- The fifth step is to multiply each value vector by the softmax score (in preparation to sum them up).
    weighted_v1 = weights[0] * v1, weighted_v2 = weights[1] * v2, ...
- Finally, we sum up the weighted value vectors to get the output vector for this word.
   z1 = weighted_v1 + weighted_v2 + weighted_v3 + ...

**Multi-Head Attention**
- Instead of performing a single self-attention calculation, the transformer architecture uses multiple self-attention calculations in parallel, known as multi-head attention. Each head has its own set of learned weight matrices (Wq, Wk, Wv) and produces its own output vector.
- The outputs of all the heads are then concatenated and linearly transformed to produce the final output vector for the word.
- Multiple heads allow the model to attend to different parts of the input sentence simultaneously, capturing different relationships and patterns in the data.
- This is particularly useful for complex tasks like language translation, where different words may have different relationships with each other.

**Positional Encoding**
- Since the transformer architecture does not have any inherent notion of word order (unlike RNNs), it uses positional encoding to inject information about the position of each word in the sentence.
- Positional encodings are added to the input embeddings at the bottom of the encoder and decoder stacks.

**The Residual Connection and Layer Normalization**
- Each sub-layer in the transformer (such as the multi-head attention layer and the feed-forward layer) has a residual connection around it, followed by layer normalization.
- This helps to stabilize the training process and allows for deeper networks.



### The decoder side of the Transformer
 - The output of the top encode block is K and V values
 - This is fed into encoder-decoder attention blocks of the decoder blocks along with the target sequence (shifted right by one position) 
 - The decoder has masked self-attention layers to prevent positions from attending to subsequent positions. This masking, combined with the fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.

 **The final linear and softmax layer**
 - The final output of the decoder stack is fed into a linear layer followed by a softmax layer to produce the final output probabilities for each token in the vocabulary.
 - The token with the highest probability is selected as the output token for that position.
 - 



## Transfromer Circuits 

Residual connections : 
 - communication channels that skip over layers
 - no computation, just copy-paste
 - help with gradient flow

Attention heads : 
 - Independent attention mechanism from each head
 - Attention heads can be thought as information routing circuits
 - where to get information from
 - what to do with the information 
 - how to combine information from different sources


Readling List
- https://jalammar.github.io/illustrated-transformer/
- http://nlp.seas.harvard.edu/2018/04/03/attention.html
- https://arxiv.org/abs/1706.03762
- Illustrated GPT-2 : https://jalammar.github.io/illustrated-gpt2/
- Krupa Dave - Everything about Transformers
- Ch3 and Ch4 of LLMs from Scratch book
- Pytorch overview 
