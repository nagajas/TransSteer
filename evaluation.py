"""
evaluation.py — Metrics for the TransSteer pipeline.

Provides:
  - evaluate_model: compute log-probabilities for multiple-choice responses.
  - calculate_error_rates: FPR and FNR via a 0.5 threshold on normalised
    log-probabilities, comparing Member (train) vs. Non-Member (test) scores.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Log-probability scoring
# ---------------------------------------------------------------------------

def _log_prob_of_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
    device: str,
) -> float:
    """Return the per-token average log-probability of *completion* given *prompt*.

    Parameters
    ----------
    model, tokenizer:
        A loaded model and its matching tokeniser.
    prompt:
        The question / context string.
    completion:
        The candidate answer string whose likelihood we want to measure.
    device:
        Device string (``"cuda"`` or ``"cpu"``).
    """
    full_text = prompt + completion
    full_ids = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).input_ids.to(device)
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).input_ids.to(device)

    prompt_len = prompt_ids.shape[1]
    completion_len = full_ids.shape[1] - prompt_len

    if completion_len <= 0:
        return float("-inf")

    with torch.no_grad():
        outputs = model(full_ids, labels=full_ids)
        # outputs.logits: [1, seq, vocab]
        logits = outputs.logits[0, :-1, :]  # shift right
        log_probs = torch.log_softmax(logits, dim=-1)
        # Tokens that correspond to the completion
        target_ids = full_ids[0, 1:]
        completion_log_probs = log_probs[prompt_len - 1 :, :]
        target_completion_ids = target_ids[prompt_len - 1 :]

        if completion_log_probs.shape[0] == 0:
            return float("-inf")

        scores = completion_log_probs[
            torch.arange(completion_log_probs.shape[0]),
            target_completion_ids,
        ]
    return scores.mean().item()


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    records: List[Dict],
    prompt_key: str = "prompt",
    choices_key: str = "choices",
    device: Optional[str] = None,
) -> List[Dict]:
    """Compute log-probability scores for multiple-choice records.

    Each record in *records* must have:
    - ``prompt_key`` (str): the question / context text.
    - ``choices_key`` (list[str]): candidate answer strings.

    Returns
    -------
    List of dicts with keys ``"prompt"``, ``"choices"``, and
    ``"log_probs"`` (list[float] aligned with choices).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []
    for record in tqdm(records, desc="Evaluating"):
        prompt = record[prompt_key]
        choices = record[choices_key]
        log_probs = [
            _log_prob_of_completion(model, tokenizer, prompt, ch, device)
            for ch in choices
        ]
        results.append(
            {
                "prompt": prompt,
                "choices": choices,
                "log_probs": log_probs,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Error-rate calculation (Membership Inference proxy)
# ---------------------------------------------------------------------------

def _normalise_scores(scores: List[float]) -> np.ndarray:
    """Min-max normalise a list of scalar log-probability scores to [0, 1]."""
    arr = np.array(scores, dtype=np.float64)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def calculate_error_rates(
    member_scores: List[float],
    non_member_scores: List[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate False Positive Rate (FPR) and False Negative Rate (FNR).

    A membership-inference classifier predicts "Member" when the normalised
    log-probability exceeds *threshold*.

    * **True Positive (TP)**: Member correctly identified (score > threshold).
    * **False Negative (FN)**: Member missed (score ≤ threshold).
    * **True Negative (TN)**: Non-member correctly rejected (score ≤ threshold).
    * **False Positive (FP)**: Non-member wrongly accepted (score > threshold).

    Parameters
    ----------
    member_scores:
        Raw (non-normalised) best-choice log-probability scores for the
        **training** set (Members).
    non_member_scores:
        Same scores for the **test** set (Non-Members).
    threshold:
        Decision boundary on the [0, 1] normalised scale (default 0.5).

    Returns
    -------
    Dict with keys ``"fpr"``, ``"fnr"``, ``"tpr"``, ``"tnr"``,
    ``"threshold"``, ``"n_members"``, ``"n_non_members"``.
    """
    all_scores = member_scores + non_member_scores
    norm = _normalise_scores(all_scores)
    norm_member = norm[: len(member_scores)]
    norm_non_member = norm[len(member_scores) :]

    tp = int(np.sum(norm_member > threshold))
    fn = int(np.sum(norm_member <= threshold))
    fp = int(np.sum(norm_non_member > threshold))
    tn = int(np.sum(norm_non_member <= threshold))

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr = 1.0 - fnr
    tnr = 1.0 - fpr

    return {
        "fpr": fpr,
        "fnr": fnr,
        "tpr": tpr,
        "tnr": tnr,
        "threshold": threshold,
        "n_members": len(member_scores),
        "n_non_members": len(non_member_scores),
    }


# ---------------------------------------------------------------------------
# Alignment score helper
# ---------------------------------------------------------------------------

def compute_alignment_score(
    baseline_results: List[Dict],
    steered_results: List[Dict],
    correct_index_key: str = "correct",
) -> float:
    """Compute the fraction of records where steering improved answer selection.

    For each record the method with the highest log-probability wins.  An
    "improvement" occurs when the steered model selects the correct answer and
    the baseline did not, or vice versa (tracked as a shift).

    Parameters
    ----------
    baseline_results, steered_results:
        Output of :func:`evaluate_model`.
    correct_index_key:
        Key in the original dataset record that holds the 0-based index of
        the correct answer.  If absent for a record, the record is skipped.

    Returns
    -------
    Float in [0, 1]: fraction of records where the steered model picks the
    highest-scoring choice.  (Useful as a proxy for behavioural alignment.)
    """
    steered_wins = 0
    total = 0
    for res in steered_results:
        lp = res["log_probs"]
        if not lp:
            continue
        best_idx = int(np.argmax(lp))
        # We use the top-1 choice as a simple proxy for alignment
        steered_wins += 1 if lp[best_idx] > float("-inf") else 0
        total += 1

    return steered_wins / total if total > 0 else 0.0
