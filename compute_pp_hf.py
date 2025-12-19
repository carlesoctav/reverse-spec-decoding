#!/usr/bin/env python3
"""
Compute perplexity of generated responses using HuggingFace transformers.

This script computes the true perplexity of responses under a given model
by running forward passes and extracting log probabilities.

Usage:
    python compute_pp_hf.py --config config/compute_pp_hf.yaml
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import math

import draccus
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


@dataclass
class PPConfig:
    """Configuration for perplexity computation."""

    model: str = "google/gemma-3-1b-it"
    input: str = "gsm8k_4b_default.jsonl"
    output: str = "gsm8k_4b_default_pp.jsonl"
    custom_chat_template: Optional[str] = None
    batch_size: int = 1  # Process one at a time for simplicity
    max_length: int = 4096
    device: str = "cuda"
    dtype: str = "bfloat16"


def load_model_and_tokenizer(cfg: PPConfig):
    """Load model and tokenizer."""
    print(f"Loading model: {cfg.model}")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(cfg.dtype, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        torch_dtype=dtype,
        device_map=cfg.device,
        attn_implementation="eager",  # For compatibility
    )
    model.eval()

    # Load custom chat template if provided
    chat_template = None
    if cfg.custom_chat_template:
        with open(cfg.custom_chat_template) as f:
            chat_template = f.read()

    return model, tokenizer, chat_template


def compute_perplexity(
    model,
    tokenizer,
    prompt: str,
    generated: str,
    chat_template: Optional[str] = None,
    max_length: int = 4096,
    device: str = "cuda",
) -> tuple[float, float, int]:
    """
    Compute perplexity of generated text given prompt.

    Returns:
        - perplexity: exp(avg negative log prob)
        - avg_logprob: average log probability of response tokens
        - num_tokens: number of response tokens
    """
    # Format the prompt
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        chat_template=chat_template,
    )

    # Full text = prompt + response
    full_text = prompt_text + generated

    # Tokenize
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    # Response tokens are the ones after the prompt
    num_prompt_tokens = len(prompt_ids)
    num_response_tokens = len(full_ids) - num_prompt_tokens

    if num_response_tokens <= 0:
        return float("inf"), float("-inf"), 0

    # Truncate if needed
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        num_response_tokens = len(full_ids) - num_prompt_tokens
        if num_response_tokens <= 0:
            return float("inf"), float("-inf"), 0

    # Convert to tensor
    input_ids = torch.tensor([full_ids], device=device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Get log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)  # [1, seq_len, vocab_size]

    # For each position i, we want the log prob of the token at position i+1
    # (because logits[i] predicts token[i+1])
    # We care about response tokens, which start at position num_prompt_tokens
    # So we need logits from positions (num_prompt_tokens-1) to (len-2)
    # to predict tokens at positions num_prompt_tokens to (len-1)

    response_logprobs = []
    for i in range(num_prompt_tokens, len(full_ids)):
        # logits at position i-1 predicts token at position i
        token_id = full_ids[i]
        log_prob = log_probs[0, i - 1, token_id].item()
        response_logprobs.append(log_prob)

    if not response_logprobs:
        return float("inf"), float("-inf"), 0

    # Compute average log probability and perplexity
    avg_logprob = sum(response_logprobs) / len(response_logprobs)
    perplexity = math.exp(-avg_logprob)

    return perplexity, avg_logprob, len(response_logprobs)


def load_records(input_path: str) -> list[dict]:
    """Load records from JSONL file."""
    records = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def get_completed_indices(output_path: str) -> set[int]:
    """Get indices already completed."""
    completed = set()
    if Path(output_path).exists():
        with open(output_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    completed.add(record["idx"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


def append_record(output_path: str, record: dict):
    """Append a record to the JSONL file."""
    with open(output_path, "a") as f:
        f.write(json.dumps(record) + "\n")


@draccus.wrap()
def main(cfg: PPConfig):
    """Main perplexity computation loop."""
    print("=" * 70)
    print("Perplexity Computation (HuggingFace)")
    print("=" * 70)
    print(f"Model: {cfg.model}")
    print(f"Template: {cfg.custom_chat_template or 'default'}")
    print(f"Input: {cfg.input}")
    print(f"Output: {cfg.output}")
    print(f"Device: {cfg.device}")
    print(f"Dtype: {cfg.dtype}")
    print("=" * 70)

    # Load model
    model, tokenizer, chat_template = load_model_and_tokenizer(cfg)
    print("Model loaded!")

    # Load input records
    print(f"\nLoading records from {cfg.input}...")
    records = load_records(cfg.input)
    print(f"Loaded {len(records)} records")

    # Get completed indices
    completed = get_completed_indices(cfg.output)
    print(f"Already completed: {len(completed)} examples")

    # Filter to pending
    pending = [r for r in records if r.get("idx") not in completed]
    print(f"To process: {len(pending)} examples")

    if not pending:
        print("Nothing to process!")
        return

    # Process records
    print("\nComputing perplexity...\n")

    total_pp = 0.0
    total_logprob = 0.0
    processed = 0
    errors = 0

    for record in tqdm(pending, desc="Processing"):
        idx = record.get("idx")
        prompt = record.get("prompt", "")
        generated = record.get("generated", "")

        try:
            pp, avg_logprob, num_tokens = compute_perplexity(
                model,
                tokenizer,
                prompt,
                generated,
                chat_template,
                cfg.max_length,
                cfg.device,
            )

            # Create output record
            out_record = {
                "idx": idx,
                "prompt": prompt,
                "generated": generated,
                "pp": pp,
                "avg_logprob": avg_logprob,
                "num_tokens": num_tokens,
            }

            # Copy additional fields
            for key in ["response", "extracted", "correct"]:
                if key in record:
                    out_record[key] = record[key]

            append_record(cfg.output, out_record)
            processed += 1
            total_pp += pp
            total_logprob += avg_logprob

        except Exception as e:
            print(f"\n[{idx}] Error: {e}")
            import traceback

            traceback.print_exc()
            errors += 1

    print("\n" + "=" * 70)
    print(f"Done! Processed: {processed}, Errors: {errors}")
    if processed > 0:
        print(f"Average Perplexity: {total_pp / processed:.4f}")
        print(f"Average Log Prob: {total_logprob / processed:.4f}")
    print(f"Output: {cfg.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
