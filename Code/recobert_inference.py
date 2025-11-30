# Source: https://github.com/r-papso/recobert 

import os

import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer

from recobert_metrics import HR_k, MPR, MRR

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PATH = os.path.join(BASE_DIR, "Data", "eval_descriptions.csv")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoint", "034_1.4295.pth")
SCORE_PATH = os.path.join(BASE_DIR, "Data", "plant_scores.pt")
LABELS_PATH = os.path.join(BASE_DIR, "Data", "plant_labels.pt")

def eval_plants():
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    df = pd.read_csv(EVAL_PATH)

    cos_sim = nn.CosineSimilarity(dim=-1)
    model = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=torch.device('cpu'))

    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = model.eval().to(device)

    prev_seed = ""
    ratings, targets = [], []

    with torch.no_grad():
        for i, seed in df.iterrows():
            if seed["seed_description"] == prev_seed:
                continue

            prev_seed = seed["seed_description"]
            rating, target = [], []

            for _, reco in df.iterrows():
                seed_d = seed["seed_description"]
                reco_t, reco_d = reco["recommended_title"], reco["recommended_description"]

                tokens = tokenizer(
                    [reco_t, reco_t],
                    [reco_d, seed_d],
                    return_special_tokens_mask=True,
                    return_token_type_ids=True,
                    padding=True,
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                special_tokens = tokens["special_tokens_mask"].to(device)
                attn_mask = tokens["attention_mask"].to(device)
                input_ids = tokens["input_ids"].to(device)
                token_types = tokens["token_type_ids"].to(device)

                out = model.forward(
                    input_ids=input_ids,
                    attn_mask=attn_mask,
                    special_tokens=special_tokens,
                    token_types=token_types,
                )

                reco_fd = out["f_d"][0:1]
                seed_fd = out["f_d"][1:2]
                dt_sim = out["cos_sim"][1:2]

                dd_sim = cos_sim.forward(seed_fd, reco_fd)

                total = sum([dt_sim, dd_sim])
                label = seed["seed_description"] == reco["seed_description"]

                rating.append(total.item())
                target.append(bool(label))

            ratings.append(rating)
            targets.append(target)

            if i > 0 and i % 10 == 0:
                print(f"Row {i} processed...")

    r = torch.tensor(ratings, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.bool)

    HR10, HR50, HR100 = HR_k(10, r, y), HR_k(50, r, y), HR_k(100, r, y)
    _MRR, _MPR = MRR(r, y), MPR(r, y)

    prnt = f"HR@10 - {HR10}, HR@50 - {HR50}, HR@100 - {HR100}, MRR: {_MRR}, MPR: {_MPR}"
    print(f"Plant dataset evaluation: {prnt}")

eval_plants()