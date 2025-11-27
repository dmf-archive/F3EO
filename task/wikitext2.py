import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, Dataset

from .base import BaseTask


def get_or_train_tokenizer(config: dict[str, Any]) -> Tokenizer:
    tokenizer_path = Path(config["data"]["tokenizer_path"])
    vocab_size = config["model"]["vocabulary_size"]

    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

        def get_training_corpus() -> Iterator[list[str]]:
            for i in range(0, len(dataset), 1000):
                batch_texts = dataset[i : i + 1000]['text']
                yield [text for text in batch_texts if text.strip()]

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, special_tokens=["<|endoftext|>", "<pad>"]
        )
        tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))

    return tokenizer


def concatenate_and_chunk(texts: list[str], tokenizer: Tokenizer, block_size: int) -> list[torch.Tensor]:
    all_tokens = []
    for text in texts:
        if text and text.strip():
            tokens = tokenizer.encode(text).ids
            all_tokens.extend(tokens)
    
    samples = []
    for i in range(0, len(all_tokens) - block_size + 1, block_size):
        chunk = all_tokens[i:i + block_size]
        samples.append(torch.tensor(chunk, dtype=torch.long))
    
    return samples


class Wikitext2Dataset(Dataset):
    def __init__(self, samples: list[torch.Tensor]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.samples[idx]
        source = seq[:-1]
        target = seq[1:]
        mask = torch.ones_like(source, dtype=torch.float)
        return {"source": source, "target": target, "mask": mask}


class Wikitext2Task(BaseTask):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.sequence_length = config["model"]["sequence_length"]
        self.batch_size = config["data"]["batch_size"]
        self.num_workers = config["data"]["num_workers"]
        self.tokenizer = get_or_train_tokenizer(config)
        self.config["model"]["vocabulary_size"] = self.tokenizer.get_vocab_size()

    def _prepare_dataset(self, split: str) -> Dataset:
        cache_dir = Path("./data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"wikitext2_{split}_concat_chunk_ids.pt"

        if cache_file.exists():
            samples = torch.load(cache_file)
        else:
            raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            all_texts = [item['text'] for item in raw_dataset if item['text'] and not item['text'].isspace()]
            samples = concatenate_and_chunk(
                all_texts,
                self.tokenizer,
                self.sequence_length + 1,
            )
            torch.save(samples, cache_file)

        return Wikitext2Dataset(samples)

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        train_dataset = self._prepare_dataset("train")
        valid_dataset = self._prepare_dataset("validation")

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=True
        )
        return train_loader, valid_loader

    def get_model(self) -> nn.Module:
        from model.nano_gpt import MiniGPT1
        model = MiniGPT1(
            vocabulary_size=self.tokenizer.get_vocab_size(),
            embedding_size=self.config["model"]["embedding_size"],
            sequence_length=self.sequence_length,
            num_heads=self.config["model"]["num_heads"],
            num_layers=self.config["model"]["num_layers"],
            learn_embeddings=True,
        )
        return model

    def get_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        hidden_weights = [
            p for n, p in model.named_parameters()
            if p.ndim >= 2 and 'transformer.h' in n
        ]
        non_hidden_weights = [
            p for n, p in model.named_parameters()
            if not (p.ndim >= 2 and 'transformer.h' in n)
        ]
        param_groups = [
            {'params': hidden_weights, 'use_diag_hadron': True, 'use_muon': True},
            {'params': non_hidden_weights, 'use_diag_hadron': False, 'use_muon': False},
        ]
        return param_groups

    @contextmanager
    def _maybe_efficient_attention(self, needs_second_order: bool):
        if needs_second_order:
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                yield
        else:
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                yield

    def train_step(self, model: nn.Module, batch: Any, criterion: nn.Module,
                   optimizer: torch.optim.Optimizer, device: torch.device,
                   needs_second_order: bool, optimizer_handles_backward: bool) -> tuple[torch.Tensor, float, dict[str, float]]:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        with self._maybe_efficient_attention(needs_second_order):
            log_probas = model(batch["source"])
            batch_size, seq_len_minus_1, vocab_size = log_probas.shape
            log_probas_flat = log_probas.view(-1, vocab_size)
            target_flat = batch["target"].view(-1)
            loss = criterion(log_probas_flat, target_flat)

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"NaN/Inf loss detected: {loss.item()}")
            if loss.item() == 0.0:
                raise RuntimeError(f"Zero loss detected: {loss.item()}")

        if not optimizer_handles_backward:
            loss.backward(create_graph=needs_second_order)
            optimizer.step()
        return log_probas.detach(), loss, {}

    def validate_epoch(self, model: nn.Module, valid_loader: DataLoader,
                       criterion: nn.Module, device: torch.device) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                log_probas = model(batch["source"])
                batch_size, seq_len_minus_1, vocab_size = log_probas.shape
                log_probas_flat = log_probas.view(-1, vocab_size)
                target_flat = batch["target"].view(-1)
                loss = criterion(log_probas_flat, target_flat)

                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError(f"NaN/Inf loss detected in validation: {loss.item()}")
                if loss.item() == 0.0:
                    raise RuntimeError(f"Zero loss detected in validation: {loss.item()}")

                total_loss += loss.item() * target_flat.numel()
                total_tokens += target_flat.numel()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss > 0 else float('inf')
        return {"loss": avg_loss, "perplexity": perplexity}
