from dataclasses import dataclass
from typing import List

from datasets import load_dataset, DatasetDict
import numpy as np
from jax import numpy as jnp
from torch.utils.data import Dataset, DataLoader
import tiktoken
from tqdm import tqdm


@dataclass
class DataSpec:
    trainloader: DataLoader
    testloader: DataLoader
    num_embeddings: int


# === tiny shakespear
class CharDataset(Dataset):

    def __init__(self, encoded: List[int], context_len: int):
        self.data = encoded
        self.context_len = context_len
    
    def __len__(self):
        return len(self.data) - self.context_len - 1
    
    def __getitem__(self, idx):
        x = np.array(self.data[idx : idx + self.context_len])
        y = np.array(self.data[idx + 1: idx + self.context_len + 1])
        return x, y


def shakespear(batch_size=16, context_size=32):
    with open('data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    # encoder: take a string, output a list of integers
    def encode(s: str) -> List[int]:
        return [stoi[c] for c in s]
    
    # decoder: take a list of integers, output a string
    def decode(x: List[int]) -> str:
        return ''.join([itos[i] for i in x]) 
    
    data = jnp.array(encode(text))
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    test_data = data[n:]
    trainloader = DataLoader(
        CharDataset(train_data, context_size), batch_size, shuffle=True
    )
    testloader = DataLoader(
        CharDataset(test_data, context_size), batch_size, shuffle=True
    )

    return DataSpec(trainloader, testloader, len(chars)), decode


# == wikitext
class TextRowDataset(Dataset):

    def __init__(self, encoded: List[List[int]], context_len: int):
        all_x = []
        all_y = []
        for row in tqdm(encoded):
            if len(row) >= context_len + 1:
                new_x, new_y = self.process_row(row, context_len)
                all_x.append(new_x)
                all_y.append(new_y)
        self.x = np.concatenate(all_x)
        self.y = np.concatenate(all_y)
        print(len(self.x))
    
    def process_row(self, row: List[int], context_len: int):
        L = len(row)
        x = np.stack([row[i: i + context_len] for i in range(L - context_len)])
        y = np.stack([row[i+1: i + context_len + 1] for i in range(L - context_len)])
        return x, y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def wikitext103(batch_size, context_len):
    data = load_dataset("wikitext", name="wikitext-103-v1")
    enc = tiktoken.get_encoding("cl100k_base")
    def encode(example):
        example["text"] = enc.encode(example["text"])
        return example

    data = data.map(encode)
    train_data = TextRowDataset(data["train"]["text"], context_len)
    test_data = TextRowDataset(data["test"]["text"], context_len)
    trainloader = DataLoader(
        train_data, batch_size, shuffle=True
    )
    testloader = DataLoader(
        test_data, batch_size, shuffle=True
    )
    
    return DataSpec(trainloader, testloader, enc.max_token_value), enc


DatasetsMap = {
    "shakespear": shakespear,
    "wiki": wikitext103,
}

# def get_batch(split: str, k):
#     print("wtf")
#     # generate a small batch of data of inputs x and targets y
#     data = train_data if split == 'train' else val_data
#     maxval = len(data) - block_size
#     idx = jax.random.randint(k, shape=(batch_size,), minval=0, maxval=maxval)
#     x = jnp.stack([data[i:i+block_size] for i in idx])
#     y = jnp.stack([data[i+1:i+block_size+1] for i in idx])
#     return x, y