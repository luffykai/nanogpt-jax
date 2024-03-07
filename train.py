from typing import List

import jax
import jax.numpy as jnp


# const
rng = jax.random.PRNGKey(123)

# hyperparam
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
# =====

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# encoder: take a string, output a list of integers
def encode(s: str) -> List[int]:
    return [stoi[c] for c in s]

# decoder: take a list of integers, output a string
def decode(x: List[int]) -> str:
    return ''.join([itos[i] for i in x]) 

data = jnp.array(encode(text), dtype=jnp.float32)
print(data.shape, data.dtype)

n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split: str, k, maxval = len(data) - block_size):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    idx = jax.random.randint(k, shape=(batch_size,), minval=0, maxval=maxval)
    x = jnp.stack([data[i:i+block_size] for i in idx])
    y = jnp.stack([data[i+1:i+block_size+1] for i in idx])
    return x, y
