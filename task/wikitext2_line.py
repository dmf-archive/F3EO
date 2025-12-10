from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from .wikitext2 import Wikitext2Dataset, Wikitext2Task


def pack_sequences_greedy(texts: list[str], tokenizer: Tokenizer, max_length: int) -> list[list[int]]:
    all_sentences = []
    for text in texts:
        if not text or not text.strip():
            continue
        sentences = text.replace('\n', ' ').split('.')
        for sent in sentences:
            sent = sent.strip()
            if sent:
                tokens = tokenizer.encode(sent + ".").ids + [tokenizer.token_to_id("<|endoftext|>")]
                if 1 < len(tokens) <= max_length:
                    all_sentences.append(tokens)

    all_sentences.sort(key=len, reverse=True)

    packs = []
    current_pack: list[int] = []
    for tokens in all_sentences:
        if len(current_pack) + len(tokens) <= max_length:
            current_pack.extend(tokens)
        else:
            if current_pack:
                packs.append(current_pack)
            current_pack = list(tokens)
    if current_pack:
        packs.append(current_pack)

    return packs

class Wikitext2LineTask(Wikitext2Task):
    def _prepare_dataset(self, split: str) -> Dataset:
        cache_dir = Path("./data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"wikitext2_{split}_line_pack_ids_v3.pt"

        if cache_file.exists():
            samples = torch.load(cache_file)
        else:
            raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            all_texts = [item['text'] for item in raw_dataset if item['text'] and not item['text'].isspace()]

            packed_sequences = pack_sequences_greedy(all_texts, self.tokenizer, self.sequence_length + 1)

            pad_token_id = self.tokenizer.token_to_id("<pad>")
            if pad_token_id is None:
                pad_token_id = 0

            samples_list = []
            for seq in packed_sequences:
                if len(seq) > self.sequence_length + 1:
                    seq = seq[:self.sequence_length + 1]

                padding_needed = (self.sequence_length + 1) - len(seq)
                padded_seq = seq + [pad_token_id] * padding_needed
                samples_list.append(torch.tensor(padded_seq, dtype=torch.long))

            if samples_list:
                samples = torch.stack(samples_list)
            else:
                samples = torch.empty(0, self.sequence_length + 1, dtype=torch.long)

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
