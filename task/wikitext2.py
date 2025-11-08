import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch.utils.data import DataLoader, Dataset


def get_or_train_tokenizer(config: dict[str, Any]) -> Tokenizer:
    """
    Loads a pre-trained tokenizer or trains a new one from the wikitext dataset.
    """
    tokenizer_path = Path(config["data"]["tokenizer_path"])
    vocab_size = config["model"]["vocabulary_size"]

    if tokenizer_path.exists():
        # Load existing tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        # Train a new tokenizer
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

        def get_training_corpus() -> Iterator[list[str]]:
            for i in range(0, len(dataset), 1000):
                # Correctly access the list of texts from the sliced dictionary
                batch_texts = dataset[i : i + 1000]['text']
                # Filter out empty or whitespace-only strings
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


class ConcatenatedWikitext2Dataset(Dataset):
    """
    A PyTorch Dataset that processes the wikitext-2 dataset by concatenating
    all articles and then chunking them into fixed-size sequences. This is
    a highly efficient method for language model pre-training.
    """
    def __init__(self, concatenated_ids: torch.Tensor, sequence_length: int):
        self.concatenated_ids = concatenated_ids
        self.sequence_length = sequence_length
        self.num_examples = (len(self.concatenated_ids) - 1) // self.sequence_length

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start_idx = idx * self.sequence_length
        seq = self.concatenated_ids[start_idx : start_idx + self.sequence_length + 1]

        source = seq[:-1]
        target = seq[1:]

        mask = torch.ones(self.sequence_length, dtype=torch.float)

        return {"source": source, "target": target, "mask": mask}


class Wikitext2Task:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.sequence_length = config["model"]["sequence_length"]
        self.batch_size = config["data"]["batch_size"]
        self.num_workers = config["data"]["num_workers"]
        self.device = config["experiment"]["device"]
        self.tokenizer = get_or_train_tokenizer(config)
        self.config["model"]["vocabulary_size"] = self.tokenizer.get_vocab_size()

    def _prepare_dataset(self, split: str) -> ConcatenatedWikitext2Dataset:
        """
        Loads pre-tokenized and concatenated data if available, otherwise
        tokenizes, concatenates, and chunks the dataset, then saves it to cache.
        """
        cache_dir = Path("./data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"wikitext2_{split}_ids.pt"

        if cache_file.exists():
            print(f"Loading cached concatenated IDs for '{split}' split...")
            concatenated_ids = torch.load(cache_file)
        else:
            print(f"No cache found. Tokenizing and concatenating '{split}' split...")
            raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

            all_token_ids = []
            eos_token_id = self.tokenizer.token_to_id("<|endoftext|>")
            if eos_token_id is None:
                eos_token_id = self.tokenizer.get_vocab_size() - 1

            for item in raw_dataset:
                text = item['text']
                if not text or text.isspace():
                    continue

                tokenized_output = self.tokenizer.encode(text)
                all_token_ids.extend(tokenized_output.ids)
                all_token_ids.append(eos_token_id)

            concatenated_ids = torch.tensor(all_token_ids, dtype=torch.long)
            print(f"Saving concatenated IDs to cache: {cache_file}")
            torch.save(concatenated_ids, cache_file)

        return ConcatenatedWikitext2Dataset(concatenated_ids, self.sequence_length)

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
        return nn.NLLLoss()

    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   monitor: Any, progress_callback=None, optimizer_tags=None) -> dict[str, float]:
        model.train()
        total_loss = 0.0
        total_tokens = 0
        last_callback_time = time.time()

        accepts_pi_signal = optimizer_tags.get("accepts_pi_signal", False) if optimizer_tags else False
        needs_second_order = optimizer_tags.get("requires_second_order", False) if optimizer_tags else False

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            optimizer.zero_grad()

            log_probas = model(batch["source"])
            loss = model.loss(log_probas, batch["target"], batch["mask"])

            loss.backward(create_graph=needs_second_order)

            # --- PI Calculation and Optimizer Step ---
            metrics = monitor.end_step(model, loss.item(), optimizer.param_groups[0]['lr'], log_probas)

            step_args = {}
            if accepts_pi_signal:
                step_args['effective_gamma'] = metrics.effective_gamma

            optimizer.step(**step_args)
            # --- End of PI Calculation ---

            total_loss += loss.item() * batch["mask"].sum().item()
            total_tokens += batch["mask"].sum().item()

            if progress_callback and (batch_idx + 1) % 10 == 0:
                current_ppl = math.exp(loss.item())
                current_time = time.time()
                time_elapsed = current_time - last_callback_time
                steps_processed = 10
                steps_per_sec = steps_processed / time_elapsed if time_elapsed > 0 else 0.0
                last_callback_time = current_time

                progress_callback(
                    epoch=monitor.current_epoch,
                    step=batch_idx + 1,
                    total_steps=len(train_loader),
                    loss=loss.item(),
                    metric=current_ppl,
                    grad_norm=metrics.grad_norm,
                    items_per_sec=steps_per_sec,
                    pi=metrics.pi,
                    effective_gamma=metrics.effective_gamma,
                    entropy=metrics.entropy
                )

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss > 0 else float('inf')

        return {"loss": avg_loss, "perplexity": perplexity}

    def validate_epoch(self, model: nn.Module, valid_loader: DataLoader,
                      criterion: nn.Module) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                log_probas = model(batch["source"])
                loss = model.loss(log_probas, batch["target"], batch["mask"])

                total_loss += loss.item() * batch["mask"].sum().item()
                total_tokens += batch["mask"].sum().item()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss > 0 else float('inf')

        return {"loss": avg_loss, "perplexity": perplexity}
