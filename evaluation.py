"""
evaluation.py — Evaluate a transformed steering vector on the target model.

Pipeline:
1. Load the target model and tokenizer.
2. Load transformed_vector.npy (produced by phase2b) and register a PyTorch
    forward hook that adds alpha * vector to the hidden states at --layer.
3. Load a corrigibility eval split.
4. For each question run two forward passes (base, steered) and compare
    log-softmax probabilities over the " A" and " B" choice tokens.
5. Write results.json with mean_accuracy, average_prob_shift, per_question_results.
"""

import argparse
import json
import os
import re
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), 'secrets.env'))
TOKEN = os.getenv("HF_TOKEN")
device = 'cuda' if torch.cuda.is_available() else 'cpu'


DATASET_CONFIGS = {
    "corrigible-more-HHH": ("corrigible-more-HHH", "corrigible-more-HHH.jsonl"),
    "myopic-reward": ("myopic-reward", "myopic-reward.jsonl"),
    "power-seeking-inclination": ("power-seeking-inclination", "power-seeking-inclination.jsonl"),
    "survival-instinct": ("survival-instinct", "survival-instinct.jsonl"),
    "wealth-seeking-inclination": ("wealth-seeking-inclination", "wealth-seeking-inclination.jsonl"),
}

CHOICE_RE = re.compile(r"\(([A-Z])\)")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
        token = TOKEN,
        quantization_config=bnb_config,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Steering hook
# ---------------------------------------------------------------------------

@contextmanager
def steering_hook(model, layer_idx: int, vector: torch.Tensor, alpha: float):
    """
    Context manager that injects `alpha * vector` into the residual stream at
    the output of the specified transformer layer.

    Works for HuggingFace models whose `.model.layers[i]` (LLaMA-style) or
    `.transformer.h[i]` (GPT-style) expose a standard module.
    """
    target_module = _get_layer_module(model, layer_idx)

    def hook_fn(module, input, output):
        # output can be a tuple (hidden, ...) or just a tensor
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        delta = (alpha * vector).to(hidden.dtype).to(hidden.device)
        hidden = hidden + delta  # broadcast over (batch, seq) dimensions

        if rest is not None:
            return (hidden,) + rest
        return hidden

    handle = target_module.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def _get_layer_module(model, layer_idx: int):
    """Return the transformer block at `layer_idx`, supporting common architectures."""
    # LLaMA / Mistral / Gemma style
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    # GPT-2 / GPT-J style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    # Falcon style
    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        return model.transformer.blocks[layer_idx]
    raise ValueError(
        f"Cannot locate transformer layers on {type(model).__name__}. "
        "Add the appropriate attribute path to _get_layer_module()."
    )


# ---------------------------------------------------------------------------
# Dataset loading + prompt formatting
# ---------------------------------------------------------------------------

def load_eval_dataset(dataset_key: str, max_samples: Optional[int]) -> list[dict]:
    """
    Returns a list of dicts with keys: prompt, label (0='A', 1='B').

    datasets' items have fields:
    question  — full MCQ text already containing " (A) ... (B) ..."
    answer_matching_behavior — "(A)" or "(B)"
    """
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_key}'. Choose from: {list(DATASET_CONFIGS)}"
        )

    hf_path, data_file = DATASET_CONFIGS[dataset_key]
    print(f"Loading dataset {hf_path} / {data_file} ...")

    ds = load_dataset("json", data_files=f"datasets/test/{data_file}", split="train")

    samples = []
    for row in ds:
        question = row.get("question", "").strip()
        answer_raw = row.get("answer_matching_behavior", "").strip()

        # Extract the letter from "(A)" / "(B)"
        m = CHOICE_RE.search(answer_raw)
        if m is None:
            continue
        label_letter = m.group(1)   # "A" or "B"
        label = 0 if label_letter == "A" else 1

        # Build the prompt: model should output a choice letter
        prompt = f"{question}\nAnswer:"
        samples.append({"prompt": prompt, "label": label})

        if max_samples is not None and len(samples) >= max_samples:
            break

    print(f"  Loaded {len(samples)} evaluation samples.")
    return samples


# ---------------------------------------------------------------------------
# Single-sample evaluation helpers
# ---------------------------------------------------------------------------

def get_choice_token_ids(tokenizer) -> tuple[int, int]:
    """
    Return token IDs for ' A' and ' B' (with leading space, as they appear
    after 'Answer:').  Falls back to 'A' / 'B' without space if needed.
    """
    for a_str, b_str in [(" A", " B"), ("A", "B"), (" (A", " (B")]:
        a_ids = tokenizer.encode(a_str, add_special_tokens=False)
        b_ids = tokenizer.encode(b_str, add_special_tokens=False)
        if len(a_ids) == 1 and len(b_ids) == 1:
            return a_ids[0], b_ids[0]
    raise RuntimeError(
        "Could not find single-token IDs for ' A' / ' B'. "
        "Inspect tokenizer.encode(' A') and ' B' manually."
    )


@torch.inference_mode()
def run_forward(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return logits for the last non-padding token of each sequence."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (batch, seq, vocab)

    # Index of the last real (non-padding) token per sequence
    seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
    last_logits = logits[torch.arange(logits.size(0)), seq_lengths]  # (batch, vocab)
    return last_logits


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model,
    tokenizer,
    samples: list[dict],
    steering_vec: torch.Tensor,
    layer_idx: int,
    alpha: float,
    batch_size: int,
    device: str,
) -> dict:
    a_id, b_id = get_choice_token_ids(tokenizer)
    print(f"Choice token IDs — A: {a_id} ('{tokenizer.decode([a_id])}'), "
        f"B: {b_id} ('{tokenizer.decode([b_id])}')")

    per_question_results = []
    correct_steered = 0

    for batch_start in tqdm(range(0, len(samples), batch_size), desc="Evaluating"):
        batch = samples[batch_start: batch_start + batch_size]
        prompts = [s["prompt"] for s in batch]
        labels = [s["label"] for s in batch]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # --- Base (no steering) forward pass ---
        base_logits = run_forward(model, input_ids, attention_mask)  # (B, vocab)
        base_lp = F.log_softmax(base_logits.float(), dim=-1)
        base_ab = base_lp[:, [a_id, b_id]]   # (B, 2)

        # --- Steered forward pass ---
        with steering_hook(model, layer_idx, steering_vec, alpha):
            steered_logits = run_forward(model, input_ids, attention_mask)
        steered_lp = F.log_softmax(steered_logits.float(), dim=-1)
        steered_ab = steered_lp[:, [a_id, b_id]]  # (B, 2)

        for i, (label, b_row, s_row) in enumerate(
            zip(labels, base_ab.tolist(), steered_ab.tolist())
        ):
            steered_choice = int(s_row[1] > s_row[0])   # 0='A', 1='B'
            is_correct = int(steered_choice == label)
            correct_steered += is_correct

            # Prob shift: change in log-prob of the correct answer
            prob_shift = s_row[label] - b_row[label]

            per_question_results.append({
                "prompt": prompts[i],
                "label": "A" if label == 0 else "B",
                "steered_choice": "A" if steered_choice == 0 else "B",
                "correct": bool(is_correct),
                "base_logprob_A": b_row[0],
                "base_logprob_B": b_row[1],
                "steered_logprob_A": s_row[0],
                "steered_logprob_B": s_row[1],
                "prob_shift": prob_shift,
            })

    n = len(per_question_results)
    mean_accuracy = correct_steered / n if n > 0 else 0.0
    average_prob_shift = (
        sum(r["prob_shift"] for r in per_question_results) / n if n > 0 else 0.0
    )

    return {
        "mean_accuracy": mean_accuracy,
        "average_prob_shift": average_prob_shift,
        "num_samples": n,
        "per_question_results": per_question_results,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load model ---
    print(f"Loading model: {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model)

    # --- Load steering vector ---
    print(f"Loading transformed vector from {args.transformed_vec} ...")
    vec_np = np.load(args.transformed_vec, allow_pickle=True)
    if vec_np.dtype == object:
        steering_vec = torch.from_numpy(vec_np.item()[args.layer]).float()  # (hidden_dim,)
    else:
        steering_vec = torch.from_numpy(vec_np).float()  # (hidden_dim,)
    # Shape for broadcasting: (1, 1, hidden_dim)
    steering_vec = steering_vec.unsqueeze(0).unsqueeze(0)
    print(f"  Steering vector shape: {steering_vec.shape}")

    # --- Load eval dataset ---
    samples = load_eval_dataset(args.dataset, args.max_samples)

    # --- Run evaluation ---
    results = evaluate(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        steering_vec=steering_vec,
        layer_idx=args.layer,
        alpha=args.alpha,
        batch_size=args.batch_size,
        device=device,
    )

    print(f"\n=== Results ===")
    print(f"  Mean accuracy   : {results['mean_accuracy']:.4f}")
    print(f"  Avg prob shift  : {results['average_prob_shift']:.4f}")
    print(f"  Num samples     : {results['num_samples']}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a phase-2b transformed steering vector on a target model."
    )
    parser.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace target model identifier.",
    )
    parser.add_argument(
        "--transformed_vec",
        default="./transformed_vector.npy",
        help="Path to transformed_vector.npy produced by phase2b.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Transformer layer index to inject the steering vector into.",
    )
    parser.add_argument(
        "--dataset",
        default="sycophancy_nlp",
        choices=list(DATASET_CONFIGS.keys()),
        help="Eval dataset split to use.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=20.0,
        help="Steering strength multiplier applied to the transformed vector.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of samples per forward pass.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap the number of eval samples (useful for quick smoke-tests).",
    )
    parser.add_argument(
        "--out",
        default="./results.json",
        help="Output path for the JSON results file.",
    )
    args = parser.parse_args()
    main(args)