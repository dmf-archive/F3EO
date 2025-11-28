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


def concatenate_and_chunk(texts: list[str], tokenizer: Tokenizer, block_size: int) -> torch.Tensor:
    all_tokens = []
    for text in texts:
        if text and text.strip():
            tokens = tokenizer.encode(text).ids
            all_tokens.extend(tokens)
    
    # Truncate to a multiple of block_size
    num_tokens = (len(all_tokens) // block_size) * block_size
    all_tokens_tensor = torch.tensor(all_tokens[:num_tokens], dtype=torch.long)
    
    return all_tokens_tensor.view(-1, block_size)


class Wikitext2Dataset(Dataset):
    def __init__(self, samples: torch.Tensor):
        self.samples = samples

    def __len__(self):
        return self.samples.size(0)

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
            if isinstance(samples, list):
                samples = torch.cat(samples)
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
        model_type = self.config["model"].get("type", "nano_gpt")
        if model_type == "rope":
            from model.qwen3_rope import Qwen3RoPEWrapper
            return Qwen3RoPEWrapper(
                vocabulary_size=self.tokenizer.get_vocab_size(),
                hidden_size=self.config["model"]["embedding_size"],
                num_hidden_layers=self.config["model"]["num_layers"],
                num_attention_heads=self.config["model"]["num_heads"],
                num_key_value_heads=self.config["model"]["num_heads"],
                max_position_embeddings=self.sequence_length,
                rope_theta=self.config["model"]["rope_theta"],
                intermediate_size=self.config["model"]["intermediate_size"],
                tie_word_embeddings=self.config["model"]["tie_word_embeddings"],
            )
        else:
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
            if p.ndim >= 2 and ('transformer.h' in n or 'model.layers' in n)
        ]
        non_hidden_weights = [
            p for n, p in model.named_parameters()
            if not (p.ndim >= 2 and ('transformer.h' in n or 'model.layers' in n))
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
            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                yield

    def train_step(self, model: nn.Module, batch: Any, criterion: nn.Module,
                   optimizer: torch.optim.Optimizer, device: torch.device,
                   needs_second_order: bool, optimizer_handles_backward: bool) -> tuple[torch.Tensor, float, dict[str, float]]:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        with self._maybe_efficient_attention(needs_second_order):
            if hasattr(model, 'loss'):
                logits = model(batch["source"])
                loss = model.loss(logits, batch["target"], batch["mask"])
                output = logits
            else:
                log_probas = model(batch["source"])
                loss = criterion(log_probas.transpose(1, 2), batch["target"])
                output = log_probas

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"NaN/Inf loss detected: {loss.item()}")
            if loss.item() == 0.0:
                raise RuntimeError(f"Zero loss detected: {loss.item()}")

        if not optimizer_handles_backward:
            loss.backward(create_graph=needs_second_order)
            optimizer.step()
        return output.detach(), loss, {}

    def validate_epoch(self, model: nn.Module, valid_loader: DataLoader,
                       criterion: nn.Module, device: torch.device) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                if hasattr(model, 'loss'):
                    logits = model(batch["source"])
                    loss = model.loss(logits, batch["target"], batch["mask"])
                    current_loss = loss.item()
                else:
                    log_probas = model(batch["source"])
                    loss = criterion(log_probas.transpose(1, 2), batch["target"])
                    current_loss = loss.item()

                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError(f"NaN/Inf loss detected in validation: {loss.item()}")
                if loss.item() == 0.0:
                    raise RuntimeError(f"Zero loss detected in validation: {loss.item()}")

                total_loss += current_loss * batch["target"].numel()
                total_tokens += batch["target"].numel()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss > 0 else float('inf')
        return {"loss": avg_loss, "perplexity": perplexity}
