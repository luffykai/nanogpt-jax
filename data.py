from dataclasses import dataclass
from typing import List

from datasets import load_dataset, DatasetDict
import numpy as np
from jax import numpy as jnp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tokenizers import Tokenizer


@dataclass
class DataSpec:
    trainloader: DataLoader
    testloader: DataLoader


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

    return DataSpec(trainloader, testloader), len(chars), decode


# == wikitext
def wikitext103(batch_size):
    data = load_dataset("wikitext", name="wikitext-103-v1")
    train_data = data["train"].filter(
        lambda example: len(example["text"]) > 0 
    )
    test_data = data["test"].filter(
        lambda example: len(example["text"]) > 0 
    )
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

    trainloader = DataLoader(
        train_data, batch_size, shuffle=True
    )
    testloader = DataLoader(
        test_data, batch_size, shuffle=True
    )
    
    return DataSpec(trainloader, testloader), tokenizer


DatasetsMap = {
    "shakespear": shakespear,
    "wiki": wikitext103,
}