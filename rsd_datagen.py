#!/usr/bin/env python3
"""
Reverse Speculative Decoding (RSD) Data Generation

Based on "In Their Own Words: Reasoning Traces Tailored for Small Models" (arXiv:2509.22230)

Algorithm: Teacher-Proposed, Student-Approved
1. Teacher proposes token y ~ P_t(·|context)
2. Student computes P_s(y) for that token
3. If P_s(y) < threshold: REJECT → Student samples its own token y ~ P_s
4. Else: ACCEPT → Keep teacher's token

This creates "student-friendly" reasoning traces with bounded surprisal.

This version uses OpenAI-compatible completions endpoints with prompt_token_ids
to avoid retokenization drift entirely. All context is passed as token IDs.
"""

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
import asyncio
import json
import math
import time

import draccus
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset
from huggingface_hub import HfApi
from openai import AsyncOpenAI
from transformers import AutoTokenizer


@dataclass
class SamplingParamsArgs:
    """Sampling parameters for generation."""

    max_tokens: int = 1
    temperature: float = 0.0
    logprobs: int = 25
    top_p: float = 1.0

    def to_oai_params(self, max_tokens: int | None = None) -> dict:
        """Create OpenAI-compatible sampling params for vLLM completions."""
        return {
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature,
            "logprobs": self.logprobs,  # For completions API, this is an int
            "top_p": self.top_p,
        }


@dataclass
class ModelEndpointArgs:
    """Arguments for an OpenAI-compatible model endpoint."""

    endpoint: str  # Base URL, e.g., "http://localhost:8000/v1"
    model: str  # Model name as registered on the server
    api_key: str = "EMPTY"  # API key (use "EMPTY" for local vLLM)

    # Optional custom chat template (path to jinja2 file to override tokenizer's default)
    custom_chat_template: Optional[str] = None

    def make_client(self) -> AsyncOpenAI:
        """Create an async OpenAI client for this endpoint."""
        return AsyncOpenAI(
            base_url=self.endpoint,
            api_key=self.api_key,
        )

    def load_chat_template(self) -> Optional[str]:
        """Load custom chat template if specified."""
        if self.custom_chat_template:
            with open(self.custom_chat_template) as f:
                return f.read()
        return None


@dataclass
class HFSource:
    """Configuration for a HuggingFace dataset source."""

    path: str
    name: Optional[str] = None
    split: str = "train"
    prompt_column: str = "question"
    response_column: str = "answer"
    is_messages: bool = False  # If True, prompt_column contains a list of messages

    def load_args(self) -> dict:
        """Get arguments for load_dataset."""
        args = {"path": self.path, "split": self.split}
        if self.name:
            args["name"] = self.name
        return args


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""

    sources: list[HFSource] = field(
        default_factory=lambda: [
            HFSource(
                path="openai/gsm8k",
                name="main",
                split="train",
                prompt_column="question",
                response_column="answer",
            )
        ]
    )
    start_idx: int = 0
    end_idx: Optional[int] = None

    def load(self) -> HFDataset:
        """Load and concatenate all dataset sources."""
        datasets = []
        for source in self.sources:
            ds = load_dataset(**source.load_args(), streaming=True)

            # Handle messages format: extract first message content as prompt
            if source.is_messages:
                prompt_col = source.prompt_column

                def extract_messages(example):
                    messages = example[prompt_col]
                    if messages and len(messages) > 0:
                        # Take the first message's content as prompt (usually user)
                        first_msg = messages[0]
                        if isinstance(first_msg, dict):
                            example["prompt"] = first_msg.get("content", "")
                        else:
                            example["prompt"] = str(first_msg)

                        # Take the second message's content as response (usually assistant)
                        if len(messages) > 1:
                            second_msg = messages[1]
                            if isinstance(second_msg, dict):
                                example["response"] = second_msg.get("content", "")
                            else:
                                example["response"] = str(second_msg)
                        else:
                            example["response"] = ""
                    else:
                        example["prompt"] = ""
                        example["response"] = ""
                    return example

                ds = ds.map(extract_messages)
                # Remove original messages column if different from "prompt"
                if source.prompt_column != "prompt":
                    ds = ds.remove_columns([source.prompt_column])
            else:
                # Rename columns to standard names
                if source.prompt_column != "prompt":
                    ds = ds.rename_column(source.prompt_column, "prompt")

                if source.response_column != "response":
                    ds = ds.rename_column(source.response_column, "response")
            datasets.append(ds)

        if len(datasets) == 1:
            return datasets[0]
        return concatenate_datasets(datasets)


@dataclass
class RSDConfig:
    """Main configuration for RSD data generation."""

    # Model endpoint configurations
    teacher: ModelEndpointArgs = field(
        default_factory=lambda: ModelEndpointArgs(
            endpoint="http://localhost:8000/v1",
            model="google/gemma-3-27b-it",
        )
    )
    student: ModelEndpointArgs = field(
        default_factory=lambda: ModelEndpointArgs(
            endpoint="http://localhost:8001/v1",
            model="google/gemma-3-1b-it",
            custom_chat_template="gemma_think.jinja",
        )
    )

    # Sampling configurations
    teacher_sampling: SamplingParamsArgs = field(
        default_factory=lambda: SamplingParamsArgs(
            max_tokens=1,
            temperature=0.0,  # Greedy for reproducibility
            logprobs=20,
        )
    )
    student_sampling: SamplingParamsArgs = field(
        default_factory=lambda: SamplingParamsArgs(
            max_tokens=1,
            temperature=0.0,  # Greedy fallback
            logprobs=20,
        )
    )

    # Dataset configuration
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # RSD parameters
    threshold: float = 0.1  # P_s(y) threshold for acceptance (10%)
    max_tokens: int = 2048  # Maximum tokens per generation
    lookahead: int = 8  # Speculative lookahead batch size
    concurrency: int = 4  # Number of examples to process concurrently

    # Output
    output: str = "rsd_traces.jsonl"

    # Hub upload
    push_to_hub: Optional[str] = None  # e.g., "username/dataset-name"


# =============================================================================
# Data Structures for Tracing
# =============================================================================


@dataclass
class GenerationStep:
    """A single step in the RSD generation process."""

    token_id: int
    token_str: str
    source: str  # "teacher" (accepted) or "student" (rejected & replaced)
    teacher_token_id: int
    teacher_token_str: str
    student_prob: float  # P_s(teacher_token)
    accepted: bool
    surprisal: float  # -log(student_prob)


@dataclass
class TraceRecord:
    """Complete trace for one example."""

    idx: int
    prompt: str
    response: str
    generated_text: str
    generated_token_ids: list[int]  # The actual token IDs for training
    steps: list
    total_tokens: int
    accepted_tokens: int
    rejected_tokens: int
    acceptance_rate: float
    avg_surprisal: float
    generation_time: float


@dataclass
class AggregateStats:
    """Aggregate statistics across all generations."""

    total_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    total_surprisal: float = 0.0
    num_examples: int = 0
    total_time: float = 0.0

    def update(self, record: TraceRecord):
        self.total_tokens += record.total_tokens
        self.accepted_tokens += record.accepted_tokens
        self.rejected_tokens += record.rejected_tokens
        self.total_surprisal += record.avg_surprisal * record.total_tokens
        self.num_examples += 1
        self.total_time += record.generation_time

    def summary(self) -> dict:
        if self.total_tokens == 0:
            return {}
        return {
            "total_tokens": self.total_tokens,
            "accepted_tokens": self.accepted_tokens,
            "rejected_tokens": self.rejected_tokens,
            "acceptance_rate": self.accepted_tokens / self.total_tokens,
            "avg_surprisal": self.total_surprisal / self.total_tokens,
            "num_examples": self.num_examples,
            "total_time": self.total_time,
            "tokens_per_second": self.total_tokens / self.total_time
            if self.total_time > 0
            else 0,
        }


# =============================================================================
# Model Clients Setup
# =============================================================================


@dataclass
class ModelPair:
    """Holds both teacher and student clients and tokenizers."""

    teacher_client: AsyncOpenAI
    student_client: AsyncOpenAI
    teacher_model: str
    student_model: str
    teacher_tokenizer: Any
    student_tokenizer: Any
    teacher_template: Optional[str]
    student_template: Optional[str]


def setup_clients(cfg: RSDConfig) -> ModelPair:
    """Setup both teacher and student OpenAI clients and tokenizers."""
    print("Setting up teacher client and tokenizer...")
    teacher_client = cfg.teacher.make_client()
    teacher_tokenizer = AutoTokenizer.from_pretrained(cfg.teacher.model)
    teacher_template = cfg.teacher.load_chat_template()

    print("Setting up student client and tokenizer...")
    student_client = cfg.student.make_client()
    student_tokenizer = AutoTokenizer.from_pretrained(cfg.student.model)
    student_template = cfg.student.load_chat_template()

    return ModelPair(
        teacher_client=teacher_client,
        student_client=student_client,
        teacher_model=cfg.teacher.model,
        student_model=cfg.student.model,
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_tokenizer,
        teacher_template=teacher_template,
        student_template=student_template,
    )


def encode_prompt(
    question: str,
    tokenizer: PreTrainedTokenizerBase,
    chat_template: Optional[str] = None,
) -> list[int]:
    """
    Encode prompt to token IDs using tokenizer's apply_chat_template.

    Args:
        question: The user question/prompt
        tokenizer: HuggingFace tokenizer
        chat_template: Optional custom Jinja2 template string to override default

    Returns:
        List of token IDs
    """
    messages = [{"role": "user", "content": question}]

    # Use apply_chat_template - handles everything including special tokens
    token_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        chat_template=chat_template,  # None = use tokenizer's default
    )

    return token_ids


# =============================================================================
# Core RSD Algorithm (OpenAI Completions API with Token IDs)
# =============================================================================


async def completion_with_token_ids(
    client: AsyncOpenAI,
    model: str,
    prompt_token_ids: list[int],
    sampling_params: dict,
) -> dict:
    """
    Generate using OpenAI completions API with token IDs as prompt.

    The OpenAI completions API natively supports passing token IDs via the
    `prompt` parameter (Iterable[int]), completely bypassing tokenization
    and avoiding retokenization drift.

    Args:
        client: AsyncOpenAI client
        model: Model name
        prompt_token_ids: List of token IDs for the prompt
        sampling_params: Sampling parameters

    Returns dict with:
        - token_ids: list of generated token IDs
        - tokens: list of generated token strings (from logprobs)
        - logprobs: list of dicts mapping token_str -> logprob
        - text: generated text
    """
    # vLLM extension: request token IDs back to avoid retokenization drift
    extra_body = {
        "return_token_ids": True,
    }

    # Pass token IDs directly via prompt parameter (native OpenAI API support)
    response = await client.completions.create(
        model=model,
        prompt=prompt_token_ids,  # OpenAI API accepts Iterable[int] directly
        extra_body=extra_body,
        **sampling_params,
    )

    choice = response.choices[0]
    result = {
        "text": choice.text,
        "token_ids": [],
        "tokens": [],
        "logprobs": [],
    }

    # Extract token_ids from vLLM's return_token_ids extension
    if hasattr(choice, "token_ids") and choice.token_ids:
        result["token_ids"] = list(choice.token_ids)

    # Extract logprobs if available
    if choice.logprobs:
        result["tokens"] = list(choice.logprobs.tokens or [])

        # Fallback: get token_ids from logprobs if not in choice
        if not result["token_ids"] and hasattr(choice.logprobs, "token_ids"):
            result["token_ids"] = list(choice.logprobs.token_ids or [])

        # top_logprobs is a list of dicts mapping token_str -> logprob
        if choice.logprobs.top_logprobs:
            result["logprobs"] = list(choice.logprobs.top_logprobs)

    return result


async def rsd_decode(
    models: ModelPair,
    prompt: str,
    cfg: RSDConfig,
    request_id_base: str,
) -> tuple[str, list[int], list[GenerationStep]]:
    """
    Reverse Speculative Decoding with batched verification.

    Uses OpenAI-compatible completions API with prompt_token_ids to completely
    avoid retokenization drift. All context is maintained as token IDs.

    Strategy:
    1. Teacher generates K tokens speculatively
    2. Student verifies all K in parallel
    3. Find first rejection, keep tokens up to there
    4. Repeat from rejection point

    Returns:
        - generated_text: The decoded text
        - generated_ids: The exact token IDs (for training)
        - steps: List of GenerationStep for analysis
    """
    BATCH_SIZE = cfg.lookahead

    # Encode prompts to token IDs locally (only done once per example)
    teacher_prompt_ids = encode_prompt(
        prompt, models.teacher_tokenizer, models.teacher_template
    )
    student_prompt_ids = encode_prompt(
        prompt, models.student_tokenizer, models.student_template
    )

    # Track generated token IDs (appended to prompts)
    generated_ids: list[int] = []
    steps: list[GenerationStep] = []

    # Get EOS token IDs for checking (both <eos> and <end_of_turn>)
    eos_token_id = models.teacher_tokenizer.eos_token_id
    end_of_turn_id = models.teacher_tokenizer.convert_tokens_to_ids("<end_of_turn>")
    stop_token_ids = {eos_token_id, end_of_turn_id}

    # Sampling params
    teacher_batch_params = cfg.teacher_sampling.to_oai_params(max_tokens=BATCH_SIZE)
    student_params = cfg.student_sampling.to_oai_params()

    def find_first_stop_token(token_ids: list[int]) -> int:
        """Find index of first stop token, or -1 if none found."""
        for i, tok_id in enumerate(token_ids):
            if tok_id in stop_token_ids:
                return i
        return -1

    while len(steps) < cfg.max_tokens:
        # === Step 1: Teacher generates batch of tokens ===
        teacher_context_ids = teacher_prompt_ids + generated_ids
        teacher_output = await completion_with_token_ids(
            models.teacher_client,
            models.teacher_model,
            teacher_context_ids,
            teacher_batch_params,
        )

        if not teacher_output["token_ids"]:
            break

        teacher_token_ids = teacher_output["token_ids"]
        teacher_tokens = teacher_output["tokens"]

        # Check for stop tokens (<eos> or <end_of_turn>)
        stop_idx = find_first_stop_token(teacher_token_ids)
        if stop_idx != -1:
            teacher_token_ids = teacher_token_ids[:stop_idx]
            teacher_tokens = teacher_tokens[:stop_idx]
            if not teacher_token_ids:
                break

        # === Step 2: Student verifies each token in parallel ===
        verify_tasks = []
        running_ids = generated_ids.copy()

        for i, tok_id in enumerate(teacher_token_ids):
            context_ids = student_prompt_ids + running_ids
            task = completion_with_token_ids(
                models.student_client,
                models.student_model,
                context_ids,
                student_params,
            )
            verify_tasks.append(task)
            running_ids.append(tok_id)  # Add teacher token for next verification

        student_outputs = await asyncio.gather(*verify_tasks)

        # === Step 3: Process results, find first rejection ===
        batch_steps = []

        for i, tok_id in enumerate(teacher_token_ids):
            tok_str = (
                teacher_tokens[i]
                if i < len(teacher_tokens)
                else models.teacher_tokenizer.decode([tok_id])
            )

            # Get student's probability for teacher's token
            student_prob = 0.0
            surprisal = 20.0

            s_output = student_outputs[i]
            if s_output and s_output["logprobs"]:
                # logprobs[0] is the logprob dict for the first (only) generated token
                token_logprobs = s_output["logprobs"][0] if s_output["logprobs"] else {}
                # Look up teacher's token string in student's top logprobs
                if tok_str in token_logprobs:
                    logprob = token_logprobs[tok_str]
                    student_prob = math.exp(logprob)
                    surprisal = -logprob

            accepted = student_prob >= cfg.threshold

            if accepted:
                # ACCEPT: Keep teacher's token
                step = GenerationStep(
                    token_id=tok_id,
                    token_str=tok_str,
                    source="teacher",
                    teacher_token_id=tok_id,
                    teacher_token_str=tok_str,
                    student_prob=student_prob,
                    accepted=True,
                    surprisal=surprisal,
                )
                batch_steps.append(step)
                generated_ids.append(tok_id)
            else:
                # REJECT: Use student's sampled token
                if s_output and s_output["token_ids"]:
                    student_tok_id = s_output["token_ids"][0]
                    student_tok_str = (
                        s_output["tokens"][0]
                        if s_output["tokens"]
                        else models.student_tokenizer.decode([student_tok_id])
                    )

                    step = GenerationStep(
                        token_id=student_tok_id,
                        token_str=student_tok_str,
                        source="student",
                        teacher_token_id=tok_id,
                        teacher_token_str=tok_str,
                        student_prob=student_prob,
                        accepted=False,
                        surprisal=surprisal,
                    )
                    batch_steps.append(step)
                    generated_ids.append(student_tok_id)
                break  # Stop at first rejection

        # === Step 4: Add batch steps ===
        steps.extend(batch_steps)

        # Check for stop tokens in generated tokens
        stop_idx = find_first_stop_token(generated_ids)
        if stop_idx != -1:
            generated_ids = generated_ids[:stop_idx]
            # Also truncate steps
            steps = steps[:stop_idx]
            break

        if not batch_steps:
            break

    # Decode final text from token IDs (for human readability only)
    generated_text = models.teacher_tokenizer.decode(generated_ids)

    # Clean up end markers in text
    end_markers = ["<end_of_turn>", "<eos>", "</s>"]
    for marker in end_markers:
        if marker in generated_text:
            generated_text = generated_text.split(marker)[0]
            break

    return generated_text, generated_ids, steps


# =============================================================================
# I/O Utilities
# =============================================================================


def get_completed_indices(output_path: str) -> set[int]:
    """Get indices already completed in output file."""
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


def append_record(output_path: str, record: TraceRecord):
    """Append a record to the JSONL file."""
    with open(output_path, "a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def load_records(output_path: str) -> list[dict]:
    """Load all records from JSONL file."""
    records = []
    if Path(output_path).exists():
        with open(output_path, "r") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except:
                    pass
    return records


# =============================================================================
# Hub Upload
# =============================================================================


def create_readme(cfg: RSDConfig, stats: dict) -> str:
    """Create README.md content for the dataset."""
    return f"""---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - rsd
  - reasoning
  - distillation
  - speculative-decoding
---

# RSD Generated Reasoning Traces

This dataset contains reasoning traces generated using **Reverse Speculative Decoding (RSD)**
from the paper ["In Their Own Words"](https://arxiv.org/abs/2509.22230).

## Generation Details

| Parameter | Value |
|-----------|-------|
| Teacher Model | `{cfg.teacher.model}` |
| Student Model | `{cfg.student.model}` |
| Acceptance Threshold | {cfg.threshold * 100:.1f}% |
| Lookahead Batch Size | {cfg.lookahead} |
| Max Tokens | {cfg.max_tokens} |

## Statistics

| Metric | Value |
|--------|-------|
| Total Examples | {stats.get("num_examples", "N/A")} |
| Total Tokens | {stats.get("total_tokens", "N/A")} |
| Teacher Tokens Accepted | {stats.get("accepted_tokens", "N/A")} ({stats.get("acceptance_rate", 0) * 100:.1f}%) |
| Teacher Tokens Rejected | {stats.get("rejected_tokens", "N/A")} ({(1 - stats.get("acceptance_rate", 0)) * 100:.1f}%) |
| Average Surprisal | {stats.get("avg_surprisal", 0):.2f} |
| Tokens/Second | {stats.get("tokens_per_second", 0):.1f} |
| Total Generation Time | {stats.get("total_time", 0) / 3600:.1f} hours |

## Algorithm

RSD is a "Teacher-Proposed, Student-Approved" generation mechanism:

1. **Teacher proposes** token `y ~ P_t(·|context)`
2. **Student verifies** by computing `P_s(y|context)`
3. If `P_s(y) < threshold`: **REJECT** → Student samples its own token
4. Else: **ACCEPT** → Keep teacher's token

This ensures the generated reasoning traces have bounded surprisal from the student's
perspective, making them more learnable during distillation.

## Dataset Format

Each record contains:
- `generated_token_ids`: Exact token IDs (no retokenization drift)
- `generated_text`: Human-readable decoded text
- `steps`: Per-token generation details

```json
{{
  "idx": 0,
  "prompt": "Janet's ducks lay 16 eggs per day...",
  "generated_text": "Let me solve this step by step...",
  "generated_token_ids": [1234, 5678, ...]
}}
```

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{cfg.push_to_hub}")

# Use token IDs directly for training (recommended)
token_ids = ds[0]["generated_token_ids"]

# Or use text
text = ds[0]["generated_text"]
```

## Citation

```bibtex
@article{{rsd2025,
  title={{In Their Own Words: Reasoning Traces Tailored for Small Models Make Them Better Reasoners}},
  year={{2025}},
  journal={{arXiv preprint arXiv:2509.22230}}
}}
```
"""


def push_to_hub(cfg: RSDConfig, stats: dict):
    """Push dataset to HuggingFace Hub."""
    if not cfg.push_to_hub:
        return

    print(f"\nPushing to Hub: {cfg.push_to_hub}")

    # Load records and create simplified dataset
    records = load_records(cfg.output)

    # Create dataset with token IDs included
    simplified = [
        {
            "idx": r["idx"],
            "prompt": r["prompt"],
            "generated_text": r["generated_text"],
            "generated_token_ids": r.get("generated_token_ids", []),
        }
        for r in records
    ]

    # Create HF dataset
    from datasets import Dataset

    ds = Dataset.from_list(simplified)

    # Create README
    readme_content = create_readme(cfg, stats)

    # Push dataset
    ds.push_to_hub(cfg.push_to_hub, private=False)

    # Upload README
    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=cfg.push_to_hub,
        repo_type="dataset",
    )

    print(f"Dataset pushed to: https://huggingface.co/datasets/{cfg.push_to_hub}")


# =============================================================================
# Main
# =============================================================================


async def async_main(cfg: RSDConfig):
    """Async main execution logic."""
    print("=" * 70)
    print("Reverse Speculative Decoding (RSD) Data Generation")
    print("(Completions API with prompt_token_ids - no retokenization drift)")
    print("=" * 70)
    print(f"Teacher: {cfg.teacher.model} @ {cfg.teacher.endpoint}")
    print(f"Student: {cfg.student.model} @ {cfg.student.endpoint}")
    print(f"Student template: {cfg.student.custom_chat_template or 'default'}")
    print(f"Threshold P_s(y): {cfg.threshold * 100:.1f}%")
    print(f"Lookahead: {cfg.lookahead} tokens")
    print(f"Max tokens: {cfg.max_tokens}")
    print(f"Output: {cfg.output}")
    if cfg.push_to_hub:
        print(f"Push to Hub: {cfg.push_to_hub}")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    dataset = cfg.dataset.load()
    print("Loaded dataset (streaming mode)")

    # Get completed indices
    completed = get_completed_indices(cfg.output)
    print(f"Already completed: {len(completed)} examples")

    # Aggregate stats (include previously completed)
    stats = AggregateStats()

    # Load stats from existing records
    if completed:
        existing_records = load_records(cfg.output)
        for r in existing_records:
            stats.total_tokens += r.get("total_tokens", 0)
            stats.accepted_tokens += r.get("accepted_tokens", 0)
            stats.rejected_tokens += r.get("rejected_tokens", 0)
            stats.total_surprisal += r.get("avg_surprisal", 0) * r.get(
                "total_tokens", 0
            )
            stats.num_examples += 1
            stats.total_time += r.get("generation_time", 0)

    # Setup clients and tokenizers
    print("\nSetting up model clients and tokenizers...")
    models = setup_clients(cfg)
    print("Ready!\n")

    # Process
    print(f"Starting RSD generation (concurrency={cfg.concurrency})...\n")

    # Semaphore to limit concurrency
    semaphore = asyncio.Semaphore(cfg.concurrency)
    # Lock for thread-safe file writes and stats updates
    write_lock = asyncio.Lock()

    async def process_example(idx: int, example: dict) -> Optional[TraceRecord]:
        """Process a single example with semaphore-limited concurrency."""
        async with semaphore:
            prompt = example["prompt"]
            response = example.get("response", "")

            print(f"[{idx}] Starting: {prompt[:50]}...")

            try:
                start_time = time.time()
                generated_text, generated_ids, steps = await rsd_decode(
                    models, prompt, cfg, f"ex_{idx}"
                )
                gen_time = time.time() - start_time

                # Stats
                accepted = sum(1 for s in steps if s.accepted)
                rejected = sum(1 for s in steps if not s.accepted)
                total = len(steps)
                acceptance_rate = accepted / total if total > 0 else 0.0
                avg_surprisal = (
                    sum(s.surprisal for s in steps) / total if total > 0 else 0.0
                )

                record = TraceRecord(
                    idx=idx,
                    prompt=prompt,
                    response=response,
                    generated_text=generated_text,
                    generated_token_ids=generated_ids,
                    steps=[asdict(s) for s in steps],
                    total_tokens=total,
                    accepted_tokens=accepted,
                    rejected_tokens=rejected,
                    acceptance_rate=acceptance_rate,
                    avg_surprisal=avg_surprisal,
                    generation_time=gen_time,
                )

                # Thread-safe write
                async with write_lock:
                    append_record(cfg.output, record)
                    stats.update(record)

                print(
                    f"[{idx}] Done: {total} tokens | "
                    f"Accept: {accepted} ({acceptance_rate * 100:.1f}%) | "
                    f"Reject: {rejected} | "
                    f"Surprisal: {avg_surprisal:.2f} | "
                    f"Time: {gen_time:.1f}s"
                )
                return record

            except Exception as e:
                print(f"[{idx}] Error: {e}")
                import traceback

                traceback.print_exc()
                return None

    # Collect examples to process
    examples_to_process = []
    idx = 0
    for example in dataset:
        # Skip to start_idx
        if idx < cfg.dataset.start_idx:
            idx += 1
            continue

        # Stop at end_idx
        if cfg.dataset.end_idx is not None and idx >= cfg.dataset.end_idx:
            break

        # Skip already completed
        if idx in completed:
            idx += 1
            continue

        examples_to_process.append((idx, example))
        idx += 1

    print(f"Collected {len(examples_to_process)} examples to process\n")

    # Process all examples concurrently
    if examples_to_process:
        tasks = [process_example(idx, ex) for idx, ex in examples_to_process]
        await asyncio.gather(*tasks)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary = stats.summary()
    if summary:
        print(f"Total examples: {summary['num_examples']}")
        print(f"Total tokens: {summary['total_tokens']}")
        print(
            f"Teacher accepted: {summary['accepted_tokens']} ({summary['acceptance_rate'] * 100:.1f}%)"
        )
        print(
            f"Teacher rejected: {summary['rejected_tokens']} ({(1 - summary['acceptance_rate']) * 100:.1f}%)"
        )
        print(f"Avg surprisal: {summary['avg_surprisal']:.2f}")
        print(f"Tokens/second: {summary['tokens_per_second']:.1f}")
        print(f"Total time: {summary['total_time'] / 60:.1f} minutes")
    print("=" * 70)

    # Push to hub
    if cfg.push_to_hub:
        push_to_hub(cfg, summary)


@draccus.wrap()
def main(cfg: RSDConfig):
    """Entry point."""
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
