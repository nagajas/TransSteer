import os
from utils import *

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import jsonlines
import random
import math
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase as Tokenizer
from transformers import PreTrainedModel as Model
from dataclasses import dataclass
from typing import Iterable
from steering_vectors import train_steering_vector, pca_aggregator
import argparse
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MWEData = list[dict[str, str]]
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STEERINGVEC_DIR = f"{CURRENT_DIR}/steering_vectors"
DATASET_DIR = f"{CURRENT_DIR}/datasets/train"

def extract_vectors(dataset, mode='mean', layers=[16,17,18,19,20], epsilon=1.0, clip=20, model_name="meta-llama/Llama-2-7B-chat-hf"):
    with jsonlines.open(f'{DATASET_DIR}/{dataset}.jsonl') as reader:
        train_data = [obj for obj in reader]
    
    random.seed(42)
    random.shuffle(train_data)

    train_dataset = make_dataset(train_data, tokenizer)

    def scaled_mean(pos, neg):
        diff = pos - neg
        scale = torch.max(torch.norm(diff, dim=1))
        diff /= scale
        return torch.mean(diff, dim=0)

    def priv_mean(pos, neg):
        diff = pos - neg
        C = clip
        n = diff.shape[0]
        norms = torch.norm(diff, dim=1)
        scale_factors = torch.clamp(C / norms, max=1.0).view(-1, 1)
        diff = diff * scale_factors
        mu = torch.mean(diff, dim=0)
        # L2-sensitivity of the clipped mean: C / n
        l2_sensitivity = C / n
        noise_std = l2_sensitivity / epsilon
        noise = torch.normal(0, noise_std, size=mu.shape).to(device)
        return mu + noise

    if mode == 'pca':
        steering = train_steering_vector(
            model,
            tokenizer,
            train_dataset,
            read_token_index=-2,
            show_progress=True,
            aggregator=pca_aggregator(),
            layers=layers
        )
    elif mode == 'private':
        steering = train_steering_vector(
            model,
            tokenizer,
            train_dataset,
            read_token_index=-2,
            show_progress=True,
            aggregator=priv_mean,
            layers=layers
        )
    else:  # mean
        steering = train_steering_vector(
            model,
            tokenizer,
            train_dataset,
            read_token_index=-2,
            show_progress=True,
            aggregator=scaled_mean,
            layers=layers
        )

    # Save per-layer vectors as a dict of numpy arrays
    vectors = {
        layer: vec.float().cpu().numpy()
        for layer, vec in steering.layer_activations.items()
    }
    
    ## save at ./steering_vectors/{model_name}/{dataset_name}/{mode}_steering_vector.npy
    os.makedirs(f"{STEERINGVEC_DIR}/{model_name}/{dataset}", exist_ok=True)
    output_path = f"{STEERINGVEC_DIR}/{model_name}/{dataset}/{mode}_steering_vector.npy"
    
    np.save(output_path, vectors)
    print(f"Saved {mode} steering vectors for layers {list(vectors.keys())} to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract steering vectors from a language model.")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-2-7B-chat-hf",
        help="HuggingFace LLM identifier"
    )
    parser.add_argument(
        "--dataset",
        default="corrigible-more-HHH",
        help="Name of dataset",
        choices=os.listdir(DATASET_DIR)
    )
    parser.add_argument(
        "--mode",
        default="mean",
        choices=["mean", "pca", "private"],
        help="Aggregation mode for steering vector extraction"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs='+',
        default=[16,17,18,19,20],
        help="Layers to extract steering vectors from"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Privacy budget (epsilon) for private mode; controls noise level (higher = less noise)"
    )
    parser.add_argument(
        "--clip",
        type=int,
        default=20,
        help="Clipping threshold for private mode"
    )
    
    args = parser.parse_args()

    model_name = args.model
    model, tokenizer = get_model_and_tokenizer(model_name, load_bfloat=True)

    extract_vectors(
        dataset=args.dataset,
        mode=args.mode,
        layers=args.layers,
        epsilon=args.epsilon,
        clip=args.clip,
        model_name=model_name.replace("/", "_")  # sanitize for file paths
    )