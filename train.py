from typing import List
import logging

from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from model import NanoGpt
from data import DatasetsMap

# TODO: setup so it's easy to switch between local run and cloud gpu run
num_epoch = 10
max_iter_per_epoch = 300
dataset_key = "wiki"
batch_size = 16
context_len = 32
n_embed = 64 # head size
n_head = 4
n_layer = 4
learning_rate = 0.001
dropout = 0.0
# =====


# TODO: should there be rng here??
def calculate_loss(state: train_state.TrainState, params, x, y, rng):
    logits = state.apply_fn(params, x, rngs = {"dropout": rng})
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss

@jax.jit
def train_step(state: train_state.TrainState, x, y, drop_rng):
    grad_fn = jax.value_and_grad(
        calculate_loss,
        argnums=1,
    )
    loss, grads = grad_fn(state, state.params, x, y, drop_rng)
    state = state.apply_gradients(grads=grads)

    return state, loss


def run_train(m: nn.Module, params, optimizer, rng, trainloader, testloader):
    state = train_state.TrainState.create(
        apply_fn=m.apply,
        params=params,
        tx=optimizer,
    )
    rng, drop_rng = jax.random.split(rng)
    for epoch in range(num_epoch):
        for batch_id, batch in enumerate(tqdm(trainloader)):
            x = jnp.array(batch[0])
            y = jnp.array(batch[1])
            state, loss = train_step(state, x, y, drop_rng)
            if batch_id > max_iter_per_epoch:
                break
        if epoch % 10 == 0 or True:
            rng, test_rng = jax.random.split(rng)
            test_batch = next(iter(testloader))
            x = jnp.array(test_batch[0])
            y = jnp.array(test_batch[1])
            val_loss = calculate_loss(state, state.params, x, y, test_rng)
            print(f"Step {epoch}: train loss: {loss}, val loss: {val_loss}")
    return state


# for shakespear only
def generate(m, rng, params, decode, max_new_tokens=100):
    def _generate(m, key, params, idx, max_new_tokens: int):
        for i in range(max_new_tokens):
            context = idx[:, -context_len:] # max context len is block_size
            key, k = jax.random.split(key)
            logits = m.apply(params, context)[:,-1,:] # at the last token, B by vocab_size
            next_idx = jax.random.categorical(k, logits)
            idx = jnp.concat([idx, jnp.expand_dims(next_idx, axis=1)], axis=1)
        return idx
    context = jnp.zeros((1, 1), dtype=jnp.int32)
    rng, generate_key = jax.random.split(rng, 2)
    idx = _generate(m, generate_key, params, context, max_new_tokens=max_new_tokens)[0]
    print(decode(idx.tolist()))


def main():
    rng = jax.random.PRNGKey(42)
    rng, init_rng, dropout_rng, data_rng = jax.random.split(rng, 4)
    logging.info(f"loading data: {dataset_key}")
    dataset, decode = DatasetsMap[dataset_key](batch_size, context_len)
    m = NanoGpt(
        num_embeddings=dataset.num_embeddings,
        n_embed=n_embed,
        context_len=context_len,
        n_layer=n_layer,
        n_head=n_head,
        training=True,
        dropout=dropout,
    )
    example_x, _ = next(iter(dataset.trainloader))  # these are np array
    params = m.init(
        {"params": init_rng, "dropout": dropout_rng},
        jnp.array(example_x),
    )
    num_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"number of params: {num_params}")
    
    optimizer = optax.adamw(learning_rate=learning_rate)

    final_state = run_train(m, params, optimizer, rng, dataset.trainloader, dataset.testloader)
    if True:
        generate(m, rng, final_state.params, decode)

if __name__ == "__main__":
    main()