import math

from flax import linen as nn
import jax
from jax import numpy as jnp


class Attention(nn.Module):
    n_embed: int  # output dim (head size)
    n_head: int
    training: bool = True
    dropout: float = 0.0

    def setup(self):
        self.kqv = nn.Dense(self.n_embed * 3, use_bias=False)
        self.attn_drop = nn.Dropout(rate=self.dropout, deterministic=not self.training)
        self.proj = nn.Dense(self.n_embed)
        self.final_drop = nn.Dropout(rate=self.dropout, deterministic=not self.training)

    def __call__(self, x):
        B, T, C = x.shape # batch size, context len, embedding dim
        # equally divided to 3 parts q k v, on the last axis
        q, k, v = jnp.split(self.kqv(x), 3, axis=2) # each shape: B, T, n_embed
        # multihead: process each n_embed in n_head parallelism
        head_size = self.n_embed // self.n_head
        q = jnp.transpose(
            jnp.reshape(q, (B, T, self.n_head, head_size)),
            axes=[0,2,1,3], # B, n_head, T, head_size
        )
        k = jnp.transpose(
            jnp.reshape(k, (B, T, self.n_head, head_size)),
            axes=[0,2,1,3],
        )
        v = jnp.transpose(
            jnp.reshape(v, (B, T, self.n_head, head_size)),
            axes=[0,2,1,3],
        )

        # B, n_head, T, T
        attn = jnp.matmul(q, jnp.transpose(k, axes=[0,1,3,2])) / math.sqrt(head_size)
        # decoder only, so triangle it
        mask = jnp.tril(jnp.ones_like(attn))
        attn = jnp.where(mask == 0, -jnp.inf, attn) # upper triangle  are all -inf
        attn = jax.nn.softmax(attn)
        attn = self.attn_drop(attn)

        y = jnp.matmul(attn, v) # B, n_head, T, n_embed
        # TODO: do we need contiguous? -> reshape order A F or C?
        y = jnp.reshape(jnp.transpose(y, [0,2,1,3]), (B, T, self.n_embed))
        y = self.final_drop(self.proj(y))

        return y


class MLP(nn.Module):
    h_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.h_dim)(x) # TODO: bias? and below too
        x = nn.gelu(x)
        x = nn.Dense(features=self.output_dim)(x)
        # TODO: dropout

        return x

class Block(nn.Module):
    n_embed: int
    bias: bool
    n_head: int
    training: bool = True
    dropout: float = 0.0

    def setup(self):
        self.ln1 = nn.LayerNorm(use_bias=self.bias)
        self.attn = Attention(n_embed=self.n_embed, n_head=self.n_head, training=self.training, dropout=self.dropout)
        self.ln2 = nn.LayerNorm(use_bias=self.bias)
        self.mlp = MLP(h_dim = 4 * self.n_embed, output_dim=self.n_embed)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class NanoGpt(nn.Module):
    vocab_size: int
    n_embed: int # embedding feature size
    block_size: int # context len
    n_layer: int
    n_head: int
    bias: bool = False # bias in linear and layernorm
    training: bool = True
    dropout: float = 0.0

    def setup(self):
        self.token_embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.n_embed)
        self.positional_embedding = nn.Embed(num_embeddings=self.block_size, features=self.n_embed)
        self.blocks = [Block(n_embed=self.n_embed, bias=self.bias, n_head=self.n_head, training=self.training, dropout=self.dropout) for _ in range(self.n_layer)]
        self.ln_f = nn.LayerNorm(use_bias=self.bias)
        self.lm_head = nn.Dense(self.vocab_size, use_bias=self.bias)

    def __call__(self, x):
        B, T = x.shape # batch size, context len
        tok_emb = self.token_embedding(x) # B, T, n_embed
        pos_emb = self.positional_embedding(jnp.arange(T)) # T, n_embed
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.lm_head(x) # B, T, vocab_size
        
        return x
    
    def generate(self, key, params, idx, max_new_tokens: int):
        self.training = False
        for _ in range(max_new_tokens):
            key, k = jax.random.split(key)
            logits = self.apply(params, idx)[:,-1,:] # at the last token, B by vocab_size
            next_idx = jax.random.categorical(k, logits)
            idx = jnp.concat([idx, jnp.expand_dims(next_idx, axis=1)], axis=1)
        
        return idx