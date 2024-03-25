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
batch_size = 32
context_len = 512
n_embed = 64 # head size
n_head = 4
n_layer = 4
learning_rate = 0.001
dropout = 0.0
# =====


def make_input(rows: List[str], tokenizer):
    encodings = tokenizer.encode_batch(rows)
    # TODO: probably not the most efficient way
    xs = []
    ys = []
    for enc in encodings:
        enc.pad(context_len+1)
        xs.append(enc.ids[:context_len])
        ys.append(enc.ids[1:(context_len+1)])
    return jnp.array(xs), jnp.array(ys)


def calculate_perplexity(state: train_state.TrainState, params, rows: List[str], tokenizer):
    pass


# TODO: dedup with the other train script
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


def train(m: nn.Module, params, optimizer, rng, trainloader, testloader, tokenizer):
    state = train_state.TrainState.create(
        apply_fn=m.apply,
        params=params,
        tx=optimizer,
    )
    rng, drop_rng = jax.random.split(rng)
    for epoch in range(num_epoch):
        for batch_id, batch in enumerate(tqdm(trainloader)):
            x, y = make_input(batch["text"], tokenizer)
            state, loss = train_step(state, x, y, drop_rng)
            if batch_id > max_iter_per_epoch:
                break
        if epoch % 10 == 0 or True:
            rng, test_rng = jax.random.split(rng)
            test_batch = next(iter(testloader))
            x, y = make_input(test_batch["text"], tokenizer)
            val_loss = calculate_loss(state, state.params, x, y, test_rng)
            print(f"Step {epoch}: train loss: {loss}, val loss: {val_loss}")
    return state


def main():
    rng = jax.random.PRNGKey(42)
    rng, init_rng, dropout_rng, data_rng = jax.random.split(rng, 4)
    logging.info(f"loading data: {dataset_key}")
    dataset, tokenizer = DatasetsMap[dataset_key](batch_size)

    m = NanoGpt(
        num_embeddings=tokenizer.get_vocab_size(),
        n_embed=n_embed,
        context_len=context_len,
        n_layer=n_layer,
        n_head=n_head,
        training=True,
        dropout=dropout,
    )
    example_batch = next(iter(dataset.trainloader))["text"]
    example_x, _ = make_input(example_batch, tokenizer)
   
    params = m.init(
        {"params": init_rng, "dropout": dropout_rng},
        jnp.array(example_x),
    )
    num_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"number of params: {num_params}")

    optimizer = optax.adamw(learning_rate=learning_rate)

    final_state = train(
        m, params, optimizer, rng, dataset.trainloader, dataset.testloader, tokenizer)

if __name__ == "__main__":
    main()