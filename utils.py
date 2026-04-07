"""
utils.py — Model & Data Loading utilities for the TransSteer pipeline.

Provides:
  - load_model: loads a HuggingFace model with 4-bit BnB quantisation.
  - unload_model: deletes a model from Python/CUDA memory.
  - DatasetLoader: reads a JSONL file and splits it 80/20 into train/test.
"""

import json
import math
import gc
import os
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(
    model_name: str,
    device_map: str = "auto",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load *model_name* in 4-bit NF4 quantisation to fit inside 4 GB of VRAM.

    Parameters
    ----------
    model_name:
        A HuggingFace Hub model identifier, e.g. ``"meta-llama/Llama-2-7b-hf"``.
    device_map:
        Passed directly to ``from_pretrained``; ``"auto"`` distributes layers
        across available devices.

    Returns
    -------
    model, tokenizer
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def unload_model(model: AutoModelForCausalLM) -> None:
    """Delete *model* from Python memory and free all GPU caches.

    Call this immediately after you are done with a model so that the next
    model can fit inside the 4 GB GPU budget.
    """
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

class DatasetLoader:
    """Read a ``.jsonl`` file and split it into train / test subsets.

    Each line of the JSONL file must be a JSON object.  The class does **not**
    impose any schema — it just stores the raw dicts so that downstream code
    can access whichever fields it needs.

    Parameters
    ----------
    path:
        Path to the ``.jsonl`` file.
    train_ratio:
        Fraction of records kept for training (default 0.8 → 80 % train,
        20 % test).
    """

    def __init__(self, path: str, train_ratio: float = 0.8) -> None:
        self.path = path
        self.train_ratio = train_ratio
        self._records: List[Dict] = []
        self._train: List[Dict] = []
        self._test: List[Dict] = []
        self._load()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))

        if not self._records:
            raise ValueError(f"Dataset file is empty: {self.path}")

        split_idx = math.ceil(len(self._records) * self.train_ratio)
        self._train = self._records[:split_idx]
        self._test = self._records[split_idx:]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def train(self) -> List[Dict]:
        """Training records (first *train_ratio* fraction)."""
        return self._train

    @property
    def test(self) -> List[Dict]:
        """Test records (remaining *1 − train_ratio* fraction)."""
        return self._test

    @property
    def all_records(self) -> List[Dict]:
        """All records in original order."""
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DatasetLoader(path={self.path!r}, "
            f"total={len(self._records)}, "
            f"train={len(self._train)}, "
            f"test={len(self._test)})"
        )
