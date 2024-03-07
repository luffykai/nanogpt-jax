from typing import List

from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from model import NanoGpt


# const
rng = jax.random.PRNGKey(123)

# hyperparam
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
n_embed = 64 # head size
n_head = 4
n_layer = 4
learning_rate = 0.001
dropout = 0.0
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

data = jnp.array(encode(text))
print(len(data))
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split: str, k):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    maxval = len(data) - block_size
    idx = jax.random.randint(k, shape=(batch_size,), minval=0, maxval=maxval)
    x = jnp.stack([data[i:i+block_size] for i in idx])
    y = jnp.stack([data[i+1:i+block_size+1] for i in idx])
    return x, y

rng, init_rng, dropout_rng, data_rng = jax.random.split(rng, 4)
example_x, example_y = get_batch("train", k=data_rng)
m = NanoGpt(
    vocab_size=vocab_size,
    n_embed=n_embed,
    block_size=block_size,
    n_layer=n_layer,
    n_head=n_head,
    training=True,
    dropout=dropout,
)
params = m.init({"params": init_rng, "dropout": dropout_rng}, example_x)
num_params = sum(x.size for x in jax.tree.leaves(params))
print(f"number of params: {num_params}")

optimizer = optax.adamw(learning_rate=learning_rate)

def calculate_loss(state: train_state.TrainState, params, batch, rng):
    x, y = batch
    logits = state.apply_fn(params, x, rngs = {"dropout": rng})
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss

@jax.jit
def train_step(state: train_state.TrainState, batch, drop_rng):
    grad_fn = jax.value_and_grad(
        calculate_loss,
        argnums=1,
    )
    loss, grads = grad_fn(state, state.params, batch, drop_rng)
    state = state.apply_gradients(grads=grads)

    return state, loss

state = train_state.TrainState.create(
    apply_fn=m.apply,
    params=params,
    tx=optimizer,
)
for epoch in range(5000):
    rng, data_key, drop_rng = jax.random.split(rng, 3)
    batch = get_batch("train", data_key)
    state, loss = train_step(state, batch, drop_rng)
    if epoch % 100 == 0:
        rng, data_key = jax.random.split(rng, 2)
        batch = get_batch("val", data_key)
        val_loss = calculate_loss(state, state.params, batch, dropout_rng)
        print(f"Step {epoch}: train loss: {loss}, val loss: {val_loss}")

def generate(m, key, params, idx, max_new_tokens: int):
    for i in range(max_new_tokens):
        context = idx[:, -block_size:] # max context len is block_size
        key, k = jax.random.split(key)
        logits = m.apply(params, context)[:,-1,:] # at the last token, B by vocab_size
        next_idx = jax.random.categorical(k, logits)
        idx = jnp.concat([idx, jnp.expand_dims(next_idx, axis=1)], axis=1)
    return idx

context = jnp.zeros((1, 1), dtype=jnp.int32)
rng, generate_key = jax.random.split(rng, 2)
idx = generate(m, generate_key, state.params, context, max_new_tokens=100)[0]
print(decode(idx.tolist()))