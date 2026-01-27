class MultiHeadAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    self.d_out = d_out
    self.d_in = d_in
    self.num_heads = num_heads
    self.context_length = context_length
    self.dropout = dropout

    assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

    self.head_dim = d_out // num_heads
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    self.out_proj = nn.Linear(d_out, d_out)

  def forward(self, X):
    batches, num_tokens, d_in = X.shape # B x L x d_in
    
    queries = self.W_query(X) # (B x L X d_in) => B x L x d_out)
    keys = self.W_key(X) # (B x L X d_in)  => B x L x d_out)
    values = self.W_value(X) # (B x L X d_in) => B x L x d_out)

    queries = queries.reshape(batches, num_tokens, self.num_heads, self.head_dim) # B x L x num_heads x head_dim
    keys = keys.reshape(batches, num_tokens, self.num_heads, self.head_dim) # B x L x num_heads x head_dim
    values = values.reshape(batches, num_tokens, self.num_heads, self.head_dim) # B x L x num_heads x head_dim

    queries = queries.transpose(1, 2) # B x num_heads x L x head_dim
    keys = keys.transpose(1, 2) # B x num_heads x L x head_dim
    values = values.transpose(1, 2) # B x num_heads x L x head_dim

    attn_scores = queries @ keys.transpose(2, 3) # (B x num_heads x L x head_dim) @ (B x num_heads x head_dim x L) => B x num_heads x L x L

    # mask : # L x L => (1 x 1 x L x L)
    attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) # B x num_heads x L x L

    attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim=-1) 
    attn_weights = nn.Dropout(self.dropout)(attn_weights) # B x num_heads x L x L

    context_vec = attn_weights @ values # (B x num_heads x L x L) @ (B x num_heads x L x head_dim)

    context_vec = context_vec.transpose(1, 2) # B x L x num_heads x head_dim
    context_vec = context_vec.reshape(batches, num_tokens, self.d_out) # B x L x d_out

    return self.out_proj(context_vec) # (B x L x d_out) @ (d_out x d_out) => B x L x d_out
