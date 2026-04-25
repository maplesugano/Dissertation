"""
Phase 1 diagnostic: linear probe on latent token hidden states.

Usage:
    python probe_latent.py --checkpoint ./ckpts/emo_coconut_simcot/checkpoint_20 \
                           --mode coconutgpt_factored  \
                           --model_id ./pretrained/gpt2 \
                           --val_path ./data/emotion_test.json \
                           --train_path ./data/emotion_train.json

Probes whether V, A, D are linearly decodable from the mean-pooled latent
representations produced by a Coconut or CoconutGPT_Factored model.
Reports R² and Pearson r per dimension for both train and test splits.
"""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer

from coconut import Coconut, CoconutGPT_Factored
from dataset import get_dataset, get_question_latent_dataset
from utils import Config, set_seed


def collect_latents(model, tokenizer, data_path, latent_id, start_id, end_id,
                    configs, device, max_samples=None, batch_size=None):
    """Run forward passes, collect mean-pooled latent vectors and VAD targets.

    Processes one sample at a time to avoid batching issues caused by
    variable-length sequences having latent tokens at different positions.
    """
    base_dataset = get_dataset(data_path, tokenizer)
    if max_samples is not None:
        base_dataset = base_dataset.select(range(min(max_samples, len(base_dataset))))

    # Use max scheduled stage for inference (full latent budget)
    dataset = get_question_latent_dataset(
        configs.max_latent_stage,
        base_dataset,
        configs,
        start_id,
        latent_id,
        end_id,
    )

    raw_data = json.load(open(data_path))
    if max_samples is not None:
        raw_data = raw_data[:max_samples]

    latent_vectors = []
    vad_targets = []

    model.eval()
    with torch.no_grad():
        for sample in tqdm(dataset, desc="Collecting latents"):
            input_ids = torch.tensor(sample["input_ids"], device=device).unsqueeze(0)
            attention_mask = torch.tensor(
                sample["attention_mask"], device=device
            ).unsqueeze(0)

            latent_positions = (input_ids[0] == latent_id).nonzero(as_tuple=True)[0]
            if len(latent_positions) == 0:
                continue

            labels = input_ids.clone()
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=device
            ).unsqueeze(0)

            outputs = model.forward(input_ids, attention_mask, labels, position_ids)
            inputs_embeds = outputs.inputs_embeds

            latent_rep = inputs_embeds[0, latent_positions, :].mean(dim=0)
            latent_vectors.append(latent_rep.cpu().float().numpy())

            d = raw_data[sample["idx"]]
            ans = d["answer"]
            if isinstance(ans, dict):
                vad_targets.append([ans["V"], ans["A"], ans["D"]])
            else:
                vad_targets.append([0.0, 0.0, 0.0])

    return np.array(latent_vectors), np.array(vad_targets)


def run_probe(X_train, y_train, X_test, y_test, alpha=1.0):
    """Fit Ridge regression probes and report per-dimension metrics."""
    dims = ["V", "A", "D"]
    results = {}
    for j, dim in enumerate(dims):
        probe = Ridge(alpha=alpha)
        probe.fit(X_train, y_train[:, j])

        pred_train = probe.predict(X_train)
        pred_test = probe.predict(X_test)

        r2_train = r2_score(y_train[:, j], pred_train)
        r2_test = r2_score(y_test[:, j], pred_test)
        r_test, _ = pearsonr(y_test[:, j], pred_test)

        results[dim] = {
            "R2_train": r2_train,
            "R2_test": r2_test,
            "Pearson_test": r_test,
        }
        print(
            f"  [{dim}]  R² train={r2_train:.4f}  R² test={r2_test:.4f}  "
            f"Pearson test={r_test:.4f}"
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", default="coconutgpt_factored",
                        choices=["coconut_baseline", "coconutgpt_factored"])
    parser.add_argument("--model_id", default="./pretrained/gpt2")
    parser.add_argument("--train_path", default="./data/emotion_train.json")
    parser.add_argument("--val_path", default="./data/emotion_test.json")
    parser.add_argument("--c_thought", type=int, default=2)
    parser.add_argument("--max_latent_stage", type=int, default=5)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build a minimal config
    config_dict = {
        "c_thought": args.c_thought,
        "max_latent_stage": args.max_latent_stage,
        "pad_latent_to_max": True,
        "mode": args.mode,
        "training_method": "full",
        "lambda_reg": 0.0,
        "lambda_con": 0.0,
        "con_temp": 1.0,
        "seed": args.seed,
    }
    configs = Config(config_dict)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    base_model = AutoModelForCausalLM.from_pretrained(args.model_id)
    base_model.resize_token_embeddings(len(tokenizer))

    if args.mode == "coconut_baseline":
        model = Coconut(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    elif args.mode == "coconutgpt_factored":
        model = CoconutGPT_Factored(
            base_model, latent_id, start_id, end_id, tokenizer.eos_token_id, configs
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print(f"Loading checkpoint from {args.checkpoint}")
    saved_weights = torch.load(args.checkpoint, map_location="cpu")
    msg = model.load_state_dict(saved_weights, strict=False)
    print(f"  missing: {len(msg.missing_keys)}  unexpected: {len(msg.unexpected_keys)}")

    model = model.to(device)
    model.eval()

    print("\nCollecting latent representations (train split) ...")
    X_train, y_train = collect_latents(
        model, tokenizer, args.train_path, latent_id, start_id, end_id,
        configs, device, max_samples=args.max_train_samples, batch_size=args.batch_size,
    )
    print(f"  collected {len(X_train)} samples, latent dim={X_train.shape[1]}")

    print("\nCollecting latent representations (test split) ...")
    X_test, y_test = collect_latents(
        model, tokenizer, args.val_path, latent_id, start_id, end_id,
        configs, device, max_samples=args.max_test_samples, batch_size=args.batch_size,
    )
    print(f"  collected {len(X_test)} samples")

    print("\n=== Linear probe results ===")
    results = run_probe(X_train, y_train, X_test, y_test, alpha=args.ridge_alpha)

    print("\nSummary:")
    for dim, m in results.items():
        print(
            f"  {dim}: Pearson={m['Pearson_test']:.4f}  "
            f"R²_test={m['R2_test']:.4f}  R²_train={m['R2_train']:.4f}"
        )


if __name__ == "__main__":
    main()
