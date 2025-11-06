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


class TokenizedWikitext2Dataset(Dataset):
    """A PyTorch Dataset for tokenized wikitext-2 data."""
    def __init__(self, split: str, tokenizer: Tokenizer, max_length: int):
        self.max_length = max_length

        # Load raw dataset
        raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        # Filter out empty lines and tokenize
        texts = [item['text'] for item in raw_dataset if item['text']]
        tokenized_output = tokenizer.encode_batch(texts)

        # Concatenate all token ids and then chunk them
        all_ids = [item.ids for item in tokenized_output]
        concatenated_ids = torch.cat([torch.tensor(ids, dtype=torch.long) for ids in all_ids])

        self.chunks = []
        for i in range(0, concatenated_ids.size(0) - max_length, max_length):
            self.chunks.append(concatenated_ids[i:i + max_length + 1])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        source = chunk[:-1]
        target = chunk[1:]
        # Mask is all ones since we don't have padding
        mask = torch.ones(self.max_length, dtype=torch.float)
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

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        train_dataset = TokenizedWikitext2Dataset("train", self.tokenizer, self.sequence_length)
        valid_dataset = TokenizedWikitext2Dataset("validation", self.tokenizer, self.sequence_length)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
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
                   monitor: Any, progress_callback=None) -> dict[str, float]:
        model.train()
        total_loss = 0.0
        total_tokens = 0
        last_callback_time = time.time()

        needs_second_order = hasattr(optimizer, '__class__') and optimizer.__class__.__name__ in ['F3EO', 'F3EL', 'F3EW', 'F3EPI', 'AdaHessian']

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            optimizer.zero_grad()
            log_probas = model(batch["source"])
            loss = model.loss(log_probas, batch["target"], batch["mask"])

            if needs_second_order:
                if optimizer.__class__.__name__ in ['F3EL', 'F3EPI']:
                    loss.backward(create_graph=True)
                    optimizer.step(loss=loss)
                else:
                    loss.backward(create_graph=True)
                    optimizer.step()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch["mask"].sum().item()
            total_tokens += batch["mask"].sum().item()

            if progress_callback and (batch_idx + 1) % 10 == 0:
                current_ppl = math.exp(loss.item())
                grad_norm = monitor.compute_grad_norm(model)
                
                # 获取F3EPI的log(PI)值
                log_pi = None
                beta_complexity = None
                if hasattr(optimizer, '__class__') and optimizer.__class__.__name__ == 'F3EPI':
                    log_pi = optimizer.last_log_pi
                    # 计算beta_complexity用于显示
                    import torch
                    beta_complexity = torch.tanh(torch.tensor(log_pi)).item() if log_pi is not None else None

                current_time = time.time()
                time_elapsed = current_time - last_callback_time
                steps_processed = 10
                steps_per_sec = steps_processed / time_elapsed if time_elapsed > 0 else 0.0
                last_callback_time = current_time

                progress_callback(batch_idx + 1, len(train_loader), loss.item(), current_ppl, grad_norm, steps_per_sec, log_pi, beta_complexity)
                
                # 更新监控器的step级别指标，包含PI值
                monitor.end_step(model, loss.item(), optimizer.param_groups[0]['lr'], log_pi, beta_complexity)

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
