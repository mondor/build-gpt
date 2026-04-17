import os
import torch
import tiktoken
import pyarrow.parquet as pq

DATA_DIR = '../../climbmix'


def list_parquet_files():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.parquet'))
    return [os.path.join(DATA_DIR, f) for f in files]


def _document_batches(split, process_rank, num_processes, batch_size=128):
    # infinite iterator over batches of text strings from parquet files
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == 'train' else parquet_paths[-1:]
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(process_rank, pf.num_row_groups, num_processes):  # step by num_processes
                rg = pf.read_row_group(rg_idx)
                texts = rg.column('text').to_pylist()
                for i in range(0, len(texts), batch_size):
                    yield texts[i: i + batch_size]


class DataLoaderClimbMix:
    def __init__(self, B, T, split, process_rank, num_processes, buffer_size=1000):
        self.B = B
        self.T = T
        self.split = split
        self.enc = tiktoken.get_encoding('gpt2')
        self.bos_token = self.enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]  # 50256
        self.row_capacity = T + 1

        self._batches = _document_batches(split, process_rank, num_processes)
        self._doc_buffer = []
        self._buffer_size = buffer_size

    def reset(self):
        pass

    def _refill_buffer(self):
        text_batch = next(self._batches)  # get another batch of 128 documents
        for text in text_batch:
            tokens = [self.bos_token] + self.enc.encode_ordinary(text)
            self._doc_buffer.append(tokens)

    def next_batch(self):
        B, T = self.B, self.T
        row_capacity = self.row_capacity
        row_buffer = torch.empty((B, row_capacity), dtype=torch.long)

        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # always keep the buffer full
                while len(self._doc_buffer) < self._buffer_size:
                    self._refill_buffer()

                remaining = row_capacity - pos

                best_idx = -1
                best_len = 0
                shortest_len = float('inf')
                shortest_idx = -1
                for i, doc in enumerate(self._doc_buffer):
                    doc_len = len(doc)
                    # find the largest document that fits entirely within the remaining
                    if doc_len <= remaining and doc_len > best_len:
                        best_len = doc_len
                        best_idx = i
                    elif doc_len < shortest_len:  # find the shortest doc
                        shortest_len = doc_len
                        shortest_idx = i

                if best_idx >= 0:
                    # Use it entirely
                    doc = self._doc_buffer.pop(best_idx)  # pop this doc
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # no doc fits - crop the shortest to fill the remaining space exactly
                    doc = self._doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        x = row_buffer[:, :-1]
        y = row_buffer[:, 1:]

        return x, y
