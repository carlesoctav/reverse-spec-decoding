#!/usr/bin/env python3
"""
Add perplexity scores to generated responses using a small model.

Usage:
    python add_pp.py --config config_pp.yaml
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import asyncio
import json
import math

import draccus
from openai import AsyncOpenAI
from transformers import AutoTokenizer


@dataclass
class ModelConfig:
    """Configuration for the evaluator model endpoint."""

    endpoint: str = "http://localhost:8001/v1"
    model: str = "google/gemma-3-1b-it"
    api_key: str = "EMPTY"
    custom_chat_template: Optional[str] = None

    def make_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(base_url=self.endpoint, api_key=self.api_key)

    def load_chat_template(self) -> Optional[str]:
        if self.custom_chat_template:
            with open(self.custom_chat_template) as f:
                return f.read()
        return None


@dataclass
class PPConfig:
    """Main configuration for perplexity calculation."""

    model: ModelConfig = field(default_factory=ModelConfig)
    input: str = "generated.jsonl"
    output: str = "generated_with_pp.jsonl"
    concurrency: int = 8
    logprobs: int = 1  # We only need logprob of the actual token


async def get_logprobs(
    client: AsyncOpenAI,
    model: str,
    prompt_ids: list[int],
    max_tokens: int = 1,
    logprobs: int = 1,
) -> dict:
    """Get logprobs for completion."""
    response = await client.completions.create(
        model=model,
        prompt=prompt_ids,
        max_tokens=max_tokens,
        temperature=0.0,
        logprobs=logprobs,
        echo=True,  # Include prompt tokens in response
    )
    return response


def encode_prompt_and_response(
    prompt: str,
    generated: str,
    tokenizer,
    chat_template: Optional[str] = None,
) -> tuple[str, str, list[int]]:
    """Encode prompt and generated response to text and token IDs."""
    messages = [{"role": "user", "content": prompt}]

    # Get the prompt text (with generation prompt but WITHOUT response)
    prompt_only = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        chat_template=chat_template,
    )

    # Full text includes the response
    full_text = prompt_only + generated

    # Encode generated text for token counting
    generated_ids = tokenizer.encode(generated, add_special_tokens=False)

    return prompt_only, full_text, generated_ids


async def calculate_perplexity(
    client: AsyncOpenAI,
    model: str,
    prompt_only: str,
    full_text: str,
    generated_ids: list[int],
    tokenizer,
) -> tuple[float, float]:
    """
    Calculate perplexity of generated text given prompt.

    Uses vLLM's prompt_logprobs parameter to get logprobs for the provided text.
    We pass the full text (prompt + response) as the prompt and request
    prompt_logprobs to get the logprobs for all tokens including the response.

    Returns:
        - perplexity: exp(avg negative log prob)
        - avg_logprob: average log probability
    """
    if not generated_ids:
        return float("inf"), float("-inf")

    # Tokenize to find where response starts
    prompt_ids = tokenizer.encode(prompt_only, add_special_tokens=False)
    num_prompt_tokens = len(prompt_ids)
    num_gen_tokens = len(generated_ids)

    # Pass full_text as prompt and use prompt_logprobs to get logprobs
    # We only need 1 new token (or 0), we care about prompt_logprobs
    response = await client.completions.create(
        model=model,
        prompt=full_text,
        max_tokens=1,
        temperature=0.0,
        extra_body={"prompt_logprobs": 1},
    )

    # Access the prompt_logprobs from response
    raw_response = response.model_dump()
    choice = raw_response.get("choices", [{}])[0]

    # prompt_logprobs is a list of dicts, one per prompt token
    # Each dict maps token_id -> logprob for top tokens
    prompt_logprobs = choice.get("prompt_logprobs", [])

    if not prompt_logprobs:
        # Fallback: if prompt_logprobs not available, return inf
        return float("inf"), float("-inf")

    # Extract logprobs for the response tokens only (skip prompt tokens)
    # prompt_logprobs[i] contains the logprob of token i given tokens 0..i-1
    # First token has no logprob (it's the start), so prompt_logprobs[0] is None
    response_logprobs = []
    for i in range(num_prompt_tokens, num_prompt_tokens + num_gen_tokens):
        if i < len(prompt_logprobs) and prompt_logprobs[i] is not None:
            # Each entry is a dict with token info
            # The actual token's logprob is in the entry
            entry = prompt_logprobs[i]
            if isinstance(entry, dict):
                # vLLM returns {token_id: {logprob: X, ...}} or similar
                # Need to get the logprob of the actual token
                for token_info in entry.values():
                    if isinstance(token_info, dict) and "logprob" in token_info:
                        response_logprobs.append(token_info["logprob"])
                        break
            elif isinstance(entry, (int, float)):
                response_logprobs.append(float(entry))

    if not response_logprobs:
        return float("inf"), float("-inf")

    # Calculate average log probability and perplexity
    avg_logprob = sum(response_logprobs) / len(response_logprobs)
    perplexity = math.exp(-avg_logprob)

    return perplexity, avg_logprob


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
            except:
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
                except:
                    pass
    return completed


def append_record(output_path: str, record: dict):
    """Append a record to the JSONL file."""
    with open(output_path, "a") as f:
        f.write(json.dumps(record) + "\n")


async def main(cfg: PPConfig):
    """Main perplexity calculation loop."""
    print("=" * 70)
    print("Perplexity Calculation")
    print("=" * 70)
    print(f"Evaluator Model: {cfg.model.model} @ {cfg.model.endpoint}")
    print(f"Template: {cfg.model.custom_chat_template or 'default'}")
    print(f"Input: {cfg.input}")
    print(f"Output: {cfg.output}")
    print(f"Concurrency: {cfg.concurrency}")
    print("=" * 70)

    # Setup client and tokenizer
    print("\nSetting up client and tokenizer...")
    client = cfg.model.make_client()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model)
    chat_template = cfg.model.load_chat_template()
    print("Ready!")

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

    # Semaphore for concurrency
    semaphore = asyncio.Semaphore(cfg.concurrency)
    write_lock = asyncio.Lock()

    processed = 0
    errors = 0
    total_pp = 0.0

    async def process_one(record: dict):
        nonlocal processed, errors, total_pp
        async with semaphore:
            idx = record.get("idx")
            prompt = record.get("prompt", "")
            generated = record.get("generated", "")

            try:
                # Encode
                prompt_only, full_text, generated_ids = encode_prompt_and_response(
                    prompt, generated, tokenizer, chat_template
                )

                # Calculate perplexity
                pp, avg_logprob = await calculate_perplexity(
                    client,
                    cfg.model.model,
                    prompt_only,
                    full_text,
                    generated_ids,
                    tokenizer,
                )

                # Create output record
                out_record = {
                    "idx": idx,
                    "prompt": prompt,
                    "generated": generated,
                    "pp": pp,
                    "avg_logprob": avg_logprob,
                    "num_tokens": len(generated_ids),
                }

                # Also copy response if present
                if "response" in record:
                    out_record["response"] = record["response"]

                async with write_lock:
                    append_record(cfg.output, out_record)
                    processed += 1
                    total_pp += pp
                    if processed % 10 == 0:
                        avg_pp = total_pp / processed
                        print(f"Processed: {processed}, Avg PP: {avg_pp:.2f}")

            except Exception as e:
                print(f"[{idx}] Error: {e}")
                import traceback

                traceback.print_exc()
                errors += 1

    print("\nCalculating perplexity...\n")

    # Process all
    tasks = [process_one(r) for r in pending]
    await asyncio.gather(*tasks)

    print("\n" + "=" * 70)
    print(f"Done! Processed: {processed}, Errors: {errors}")
    if processed > 0:
        print(f"Average Perplexity: {total_pp / processed:.2f}")
    print(f"Output: {cfg.output}")
    print("=" * 70)


@draccus.wrap()
def entry(cfg: PPConfig):
    asyncio.run(main(cfg))


if __name__ == "__main__":
    entry()
