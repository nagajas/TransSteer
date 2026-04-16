import argparse
import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), 'secrets.env'))
TOKEN = os.getenv("HF_TOKEN")
device = "cuda" if torch.cuda.is_available() else "cpu"

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


def collect_samples(tokenizer: AutoTokenizer, num_samples: int = 500, max_length: int = 128) -> list[str]:
    """Load and filter wikitext-2-raw-v1, returning up to num_samples non-empty texts."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = []
    for row in dataset:
        text = row["text"].strip()
        if not text:
            continue
        texts.append(text)
        if len(texts) >= num_samples:
            break
    return texts


def dump_anchors(
    model_name: str,
    layer: int,
    out_file: str,
    batch_size: int = 16,
    max_length: int = 128,
    num_samples: int = 500,
):
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name)

    print(f"Loading {num_samples} samples from wikitext-2-raw-v1 ...")
    texts = collect_samples(tokenizer, num_samples=num_samples, max_length=max_length)
    print(f"Collected {len(texts)} samples")

    all_activations: list[np.ndarray] = []

    for batch_start in tqdm(range(0, len(texts), batch_size), desc="Extracting activations"):
        batch_texts = texts[batch_start: batch_start + batch_size]

        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # hidden_states is a tuple of (num_layers + 1) tensors, each (batch, seq, hidden)
        # index 0 is the embedding layer; layer i corresponds to index i
        hidden = outputs.hidden_states[layer]  # (batch, seq_len, hidden_dim)

        # Mean-pool over non-padding tokens to get one vector per sample
        mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        summed = (hidden.float() * mask).sum(dim=1)   # (batch, hidden_dim)
        counts = mask.sum(dim=1)                       # (batch, 1)
        pooled = (summed / counts).cpu().numpy()       # (batch, hidden_dim)

        all_activations.append(pooled)

    activation_matrix = np.concatenate(all_activations, axis=0)  # (num_samples, hidden_dim)
    print(f"Activation matrix shape: {activation_matrix.shape}")

    out_path = os.path.join(os.path.dirname(__file__), "anchors")
    os.makedirs(out_path, exist_ok=True)
    
    out_file = os.path.join(out_path, f"{model_name.replace('/', '_')}_layer{layer}_anchors.npy")
    np.save(out_file, activation_matrix)
    print(f"Saved activations to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dump anchor activations from a model layer using wikitext-2-raw-v1."
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-2-7B-chat-hf",
        help="HuggingFace model identifier (e.g. meta-llama/Llama-3.1-8B or a 1B-3B target model)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=16,
        help="Transformer layer index to extract hidden states from (0 = embedding layer)",
    )
    parser.add_argument(
        "--out_file",
        default="./anchors.npy",
        help="Output path for the activation matrix .npy file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for forward passes",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum token length to truncate samples to",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of wikitext samples to use",
    )
    args = parser.parse_args()

    dump_anchors(
        model_name=args.model,
        layer=args.layer,
        out_file=args.out_file,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_samples=args.num_samples,
    )