# Source: https://github.com/r-papso/recobert

import os

import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import (
    BertModel,
    BertTokenizer,
    DataCollatorForLanguageModeling
)

from recobert_data import CollatorWrapper, RecoDataset
from recobert_model import RecoBERT
from recobert_utils import train

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Data", "generated_descriptions.csv")

if __name__ == "__main__":
    # Define hyper-parameters
    lr = 0.0001
    l2_reg = 0.0
    beta1 = 0.9
    beta2 = 0.999
    epochs = 100
    early_stop = 100
    batch_size = 32
    workers = 8

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    bert = BertModel.from_pretrained("bert-base-cased")
    collator = DataCollatorForLanguageModeling(tokenizer)
    collate_fn = CollatorWrapper(tokenizer, collator)

    df = pd.read_csv(DATA_PATH)
    dataset = RecoDataset(df=df, swap_prob=0.5)

    # Dataset split
    idxs = list(range(len(dataset)))
    split = int(len(dataset) * 0.8)
    train_idxs, val_idxs = idxs[:split], idxs[split:]

    train_sampler = SubsetRandomSampler(train_idxs)
    val_sampler = SubsetRandomSampler(val_idxs)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=workers,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=workers,
    )

    model = RecoBERT(bert, tokenizer.vocab_size)

    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = model.to(device)

    optim = Adam(model.parameters(), lr=lr, weight_decay=l2_reg, betas=(beta1, beta2))

    model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=optim,
        epochs=epochs,
        device=device,
        checkpoint="./checkpoint",
        early_stop=early_stop,
    )