# dataset.py

import torch
from torch.utils.data import IterableDataset
import tiktoken


class StreamingLanguageModelDataset(IterableDataset):
    def __init__(self, iterable_ds, seq_len, tokenizer, max_tokens=None):
        super().__init__()
        self.iterable_ds = iterable_ds
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.generated_tokens = 0

    def __iter__(self):
        token_buffer = []

        for item in self.iterable_ds:
            if self.max_tokens is not None and self.generated_tokens >= self.max_tokens:
                break

            text = item.get("text", "") # Use get to be safe
            # Tokenizers library encode returns an Encoding object, we need .ids
            tokens = self.tokenizer.encode(text).ids
            token_buffer.extend(tokens)

            while len(token_buffer) >= self.seq_len + 1:
                chunk = token_buffer[: self.seq_len + 1]
                token_buffer = token_buffer[self.seq_len + 1 :]

                chunk = torch.tensor(chunk, dtype=torch.long)
            
                self.generated_tokens += self.seq_len

                yield {
                    "input_ids": chunk[:-1],
                    "targets": chunk[1:]
                }
