from __future__ import annotations

from typing import Iterable

import torch
from datasets import DatasetDict, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from .config import ExperimentConfig


def resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset(config: ExperimentConfig):
    return load_from_disk(str(config.dataset_path))


def build_tokenizer(config: ExperimentConfig):
    return AutoTokenizer.from_pretrained(config.model_checkpoint, model_max_length=config.max_length)


def tokenize_dataset(dataset, tokenizer, remove_text: bool = False):
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized = dataset.map(tokenize_function, load_from_cache_file=True)
    if remove_text and "text" in tokenized.column_names[next(iter(tokenized.keys()))]:
        tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format("torch")
    return tokenized


def build_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer)


def build_dataloader(dataset_split, collator, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset_split, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


def maybe_remove_columns(dataset_split, columns: Iterable[str]):
    existing = [column for column in columns if column in dataset_split.column_names]
    return dataset_split.remove_columns(existing) if existing else dataset_split
