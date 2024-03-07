import math

import flax
from flax import linen as nn
import jax
from jax import numpy as jnp


class Attention(nn.Module):
    n_embed: int  # output dim (head size)
    n_head: int # TODO: multi-head
    use_bias: bool = False

    def setup(self):
        self.kqv = nn.Dense(self.n_embed * 3, use_bias=self.use_bias)

    def __call__(self, x):
        B, T, C = x.shape # batch size, context len, embedding dim
        # equally divided to 3 parts q k v, on the last axis
        q, k, v = jnp.split(self.kqv(x), 3, axis=2) # each shape: B, T, n_embed

        attn = jnp.matmul(q, jnp.transpose(k, axes=[0,2,1])) / math.sqrt(self.n_embed)
        # decoder only, so triangle it
        mask = jnp.tril(jnp.ones_like(attn))
        attn = jnp.where(mask == 0, -jnp.inf, attn) # upper triangle  are all -inf
        attn = jax.nn.softmax(attn) # B, T, T

        y = jnp.matmul(attn, v) # B, T, n_embed

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

    def setup(self):
        self.ln1 = nn.LayerNorm()
        self.attn = Attention(n_embed=self.n_embed, n_head=1)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        return x

class NanoGpt(nn.Module):
    n_block: int

    def setup(self):
        self.blocks = [Block(n_embed=16) for _ in range(self.n_block)]

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        
        return x