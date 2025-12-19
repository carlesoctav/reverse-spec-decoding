#!/usr/bin/env python3
"""
Generate responses from a single model.

Usage:
    python generate.py --config config_generate.yaml
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import asyncio
import json

import draccus
from datasets import load_dataset
from openai import AsyncOpenAI
from transformers import AutoTokenizer


@dataclass
class ModelConfig:
    """Configuration for the model endpoint."""

    endpoint: str = "http://localhost:8000/v1"
    model: str = "google/gemma-3-4b-it"
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
class SamplingConfig:
    """Sampling parameters."""

    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    path: str = "openai/gsm8k"
    name: Optional[str] = "main"
    split: str = "train"
    prompt_column: str = "question"
    response_column: str = "answer"
    is_messages: bool = False
    start_idx: int = 0
    end_idx: Optional[int] = None


@dataclass
class GenerateConfig:
    """Main configuration for generation."""

    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: str = "generated.jsonl"
    concurrency: int = 8


def load_data(cfg: DatasetConfig):
    """Load dataset."""
    args = {"path": cfg.path, "split": cfg.split, "streaming": True}
    if cfg.name:
        args["name"] = cfg.name

    ds = load_dataset(**args)
    return ds


def encode_prompt(
    question: str,
    tokenizer,
    chat_template: Optional[str] = None,
) -> list[int]:
    """Encode prompt to token IDs."""
    messages = [{"role": "user", "content": question}]
    token_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        chat_template=chat_template,
    )
    return token_ids


async def generate_one(
    client: AsyncOpenAI,
    model: str,
    prompt_ids: list[int],
    sampling: SamplingConfig,
) -> str:
    """Generate response for one prompt."""
    response = await client.completions.create(
        model=model,
        prompt=prompt_ids,
        max_tokens=sampling.max_tokens,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
    )
    return response.choices[0].text


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


async def main(cfg: GenerateConfig):
    """Main generation loop."""
    print("=" * 70)
    print("Single Model Generation")
    print("=" * 70)
    print(f"Model: {cfg.model.model} @ {cfg.model.endpoint}")
    print(f"Template: {cfg.model.custom_chat_template or 'default'}")
    print(f"Max tokens: {cfg.sampling.max_tokens}")
    print(f"Temperature: {cfg.sampling.temperature}")
    print(f"Concurrency: {cfg.concurrency}")
    print(f"Output: {cfg.output}")
    print("=" * 70)

    # Setup client and tokenizer
    print("\nSetting up client and tokenizer...")
    client = cfg.model.make_client()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model)
    chat_template = cfg.model.load_chat_template()
    print("Ready!")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_data(cfg.dataset)
    print("Dataset loaded (streaming)")

    # Get completed indices
    completed = get_completed_indices(cfg.output)
    print(f"Already completed: {len(completed)} examples")

    # Semaphore for concurrency
    semaphore = asyncio.Semaphore(cfg.concurrency)
    write_lock = asyncio.Lock()

    processed = 0
    errors = 0

    async def process_one(idx: int, example: dict):
        nonlocal processed, errors
        async with semaphore:
            # Extract prompt
            if cfg.dataset.is_messages:
                messages = example[cfg.dataset.prompt_column]
                if messages and len(messages) > 0:
                    first_msg = messages[0]
                    prompt = (
                        first_msg.get("content", "")
                        if isinstance(first_msg, dict)
                        else str(first_msg)
                    )
                else:
                    prompt = ""
            else:
                prompt = example[cfg.dataset.prompt_column]

            # Extract response (ground truth)
            if cfg.dataset.is_messages:
                messages = example[cfg.dataset.prompt_column]
                if messages and len(messages) > 1:
                    second_msg = messages[1]
                    response = (
                        second_msg.get("content", "")
                        if isinstance(second_msg, dict)
                        else str(second_msg)
                    )
                else:
                    response = ""
            else:
                response = example.get(cfg.dataset.response_column, "")

            try:
                # Encode prompt
                prompt_ids = encode_prompt(prompt, tokenizer, chat_template)

                # Generate
                generated = await generate_one(
                    client, cfg.model.model, prompt_ids, cfg.sampling
                )

                # Clean up
                end_markers = ["<end_of_turn>", "<eos>", "</s>"]
                for marker in end_markers:
                    if marker in generated:
                        generated = generated.split(marker)[0]
                        break

                record = {
                    "idx": idx,
                    "prompt": prompt,
                    "response": response,
                    "generated": generated.strip(),
                }

                async with write_lock:
                    append_record(cfg.output, record)
                    processed += 1
                    if processed % 10 == 0:
                        print(f"Processed: {processed}")

            except Exception as e:
                print(f"[{idx}] Error: {e}")
                errors += 1

    # Collect examples
    examples = []
    idx = 0
    for example in dataset:
        if idx < cfg.dataset.start_idx:
            idx += 1
            continue
        if cfg.dataset.end_idx is not None and idx >= cfg.dataset.end_idx:
            break
        if idx in completed:
            idx += 1
            continue
        examples.append((idx, example))
        idx += 1

    print(f"\nExamples to process: {len(examples)}")
    print("Starting generation...\n")

    # Process all
    tasks = [process_one(idx, ex) for idx, ex in examples]
    await asyncio.gather(*tasks)

    print("\n" + "=" * 70)
    print(f"Done! Processed: {processed}, Errors: {errors}")
    print(f"Output: {cfg.output}")
    print("=" * 70)


@draccus.wrap()
def entry(cfg: GenerateConfig):
    asyncio.run(main(cfg))


if __name__ == "__main__":
    entry()
