"""
extraction.py - Steering Vector Extraction Script

This script extracts steering vectors from Llama-2-7B using 4-bit quantization.
It supports both Standard Mean and Private Mean (with differential privacy) extraction methods.
"""

import os
import gc
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from steering_vectors import train_steering_vector
from steering_vectors.aggregators import mean_aggregator, private_mean_aggregator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract steering vectors from Llama-2-7B"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./vectors",
        help="Directory to save extracted vectors",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to extract vectors from (default: all layers)",
    )
    parser.add_argument(
        "--read_token_index",
        type=int,
        default=-2,
        help="Token index to read activations from",
    )
    parser.add_argument(
        "--noise_multiplier",
        type=float,
        default=1.0,
        help="Noise multiplier for private mean aggregator (differential privacy)",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=1.0,
        help="Clipping bound for private mean aggregator (differential privacy)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to training dataset for steering vector extraction",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str):
    """
    Load Llama-2-7B model in 4-bit quantization using BitsAndBytesConfig.
    
    Args:
        model_name: HuggingFace model name or path
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model: {model_name} with 4-bit quantization...")
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    print("Model loaded successfully!")
    return model, tokenizer


def load_dataset(dataset_path: str = None):
    """
    Load or create a training dataset for steering vector extraction.
    
    Args:
        dataset_path: Optional path to a dataset file
        
    Returns:
        Training dataset suitable for steering vector extraction
    """
    if dataset_path and os.path.exists(dataset_path):
        # Load dataset from file (assuming it's a pickle or similar format)
        import pickle
        with open(dataset_path, "rb") as f:
            train_dataset = pickle.load(f)
        print(f"Loaded dataset from {dataset_path}")
    else:
        # Create a minimal example dataset
        # Users should provide their own dataset for actual use
        print("Warning: No dataset provided. Using minimal example dataset.")
        print("Please provide a dataset via --dataset_path for actual extraction.")
        train_dataset = []
    
    return train_dataset


def extract_standard_mean_vector(
    model,
    tokenizer,
    train_dataset,
    read_token_index: int = -2,
    layers=None,
):
    """
    Extract steering vector using Standard Mean aggregation.
    
    Args:
        model: The loaded language model
        tokenizer: The tokenizer
        train_dataset: Training dataset for extraction
        read_token_index: Token index to read activations from
        layers: Specific layers to extract from
        
    Returns:
        Steering vector using standard mean aggregation
    """
    print("Extracting steering vector with Standard Mean aggregator...")
    
    steering_vector = train_steering_vector(
        model,
        tokenizer,
        train_dataset,
        read_token_index=read_token_index,
        show_progress=True,
        aggregator=mean_aggregator,
        layers=layers,
    )
    
    print("Standard Mean steering vector extracted successfully!")
    return steering_vector


def extract_private_mean_vector(
    model,
    tokenizer,
    train_dataset,
    read_token_index: int = -2,
    layers=None,
    noise_multiplier: float = 1.0,
    clip: float = 1.0,
):
    """
    Extract steering vector using Private Mean aggregation with differential privacy.
    
    Args:
        model: The loaded language model
        tokenizer: The tokenizer
        train_dataset: Training dataset for extraction
        read_token_index: Token index to read activations from
        layers: Specific layers to extract from
        noise_multiplier: Noise multiplier for differential privacy
        clip: Clipping bound for differential privacy
        
    Returns:
        Steering vector using private mean aggregation
    """
    print(f"Extracting steering vector with Private Mean aggregator...")
    print(f"  noise_multiplier: {noise_multiplier}")
    print(f"  clip: {clip}")
    
    # Create private mean aggregator with differential privacy parameters
    aggregator = private_mean_aggregator(
        noise_multiplier=noise_multiplier,
        clip=clip,
    )
    
    steering_vector = train_steering_vector(
        model,
        tokenizer,
        train_dataset,
        read_token_index=read_token_index,
        show_progress=True,
        aggregator=aggregator,
        layers=layers,
    )
    
    print("Private Mean steering vector extracted successfully!")
    return steering_vector


def save_vectors(standard_vector, private_vector, output_dir: str):
    """
    Save the extracted steering vectors as .npy files.
    
    Args:
        standard_vector: Steering vector from standard mean aggregation
        private_vector: Steering vector from private mean aggregation
        output_dir: Directory to save the vectors
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save standard mean vector
    standard_path = os.path.join(output_dir, "standard_mean_vector.npy")
    np.save(standard_path, standard_vector)
    print(f"Standard Mean steering vector saved to {standard_path}")
    
    # Save private mean vector
    private_path = os.path.join(output_dir, "private_mean_vector.npy")
    np.save(private_path, private_vector)
    print(f"Private Mean steering vector saved to {private_path}")


def purge_vram(model, tokenizer):
    """
    Delete model and tokenizer and empty the CUDA cache to free VRAM.
    
    Args:
        model: The loaded model to delete
        tokenizer: The tokenizer to delete
    """
    print("Purging VRAM...")
    
    # Delete model and tokenizer
    del model
    del tokenizer
    
    # Run garbage collection
    gc.collect()
    
    # Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"CUDA memory after purge: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")
    
    print("VRAM purged successfully!")


def main():
    """Main function to run the steering vector extraction pipeline."""
    args = parse_args()
    
    print("=" * 60)
    print("Steering Vector Extraction Pipeline")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    try:
        # Load training dataset
        train_dataset = load_dataset(args.dataset_path)
        
        if len(train_dataset) == 0:
            print("Error: Empty dataset. Please provide a valid dataset.")
            print("Skipping vector extraction.")
            standard_vector = None
            private_vector = None
        else:
            # Extract Standard Mean steering vector
            standard_vector = extract_standard_mean_vector(
                model,
                tokenizer,
                train_dataset,
                read_token_index=args.read_token_index,
                layers=args.layers,
            )
            
            # Extract Private Mean steering vector
            private_vector = extract_private_mean_vector(
                model,
                tokenizer,
                train_dataset,
                read_token_index=args.read_token_index,
                layers=args.layers,
                noise_multiplier=args.noise_multiplier,
                clip=args.clip,
            )
            
            # Save the extracted vectors
            save_vectors(standard_vector, private_vector, args.output_dir)
    
    finally:
        # Always purge VRAM at the end
        purge_vram(model, tokenizer)
    
    print("=" * 60)
    print("Extraction pipeline completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
