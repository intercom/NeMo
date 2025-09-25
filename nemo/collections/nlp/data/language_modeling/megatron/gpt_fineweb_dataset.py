# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FineWeb dataset for GPT-style pre-training using HuggingFace datasets."""

import math
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.classes import Dataset
from nemo.utils import logging

__all__ = ['GPTFineWebDataset']


class GPTFineWebDataset(Dataset):
    """
    Dataset for loading FineWeb data from HuggingFace for GPT-style pre-training.

    This dataset handles the FineWeb dataset which contains web-crawled text data
    with a 'text' feature in the 'train' split.
    """

    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb",
        subset: Optional[str] = None,
        tokenizer: TokenizerSpec = None,
        max_seq_length: int = 2048,
        min_seq_length: int = 1,
        max_num_samples: Optional[int] = None,
        seed: int = 1234,
        add_bos: bool = True,
        add_eos: bool = True,
        streaming: bool = True,
        cache_dir: Optional[str] = None,
        num_proc: Optional[int] = None,
        split: str = "train",
        text_field: str = "text",
        pack_sequences: bool = True,
        pad_to_max_length: bool = False,
    ):
        """
        Initialize the FineWeb dataset.

        Args:
            dataset_name: HuggingFace dataset name (default: "HuggingFaceFW/fineweb")
            subset: Specific subset of FineWeb to use (e.g., "sample-10BT")
            tokenizer: Tokenizer for the dataset
            max_seq_length: Maximum sequence length for each example
            min_seq_length: Minimum sequence length for valid examples
            max_num_samples: Maximum number of samples to use (None for unlimited)
            seed: Random seed for shuffling
            add_bos: Whether to add beginning of sequence token
            add_eos: Whether to add end of sequence token
            streaming: Whether to use streaming mode (recommended for large datasets)
            cache_dir: Directory to cache the dataset
            num_proc: Number of processes for dataset loading
            split: Dataset split to use (default: "train")
            text_field: Field name containing the text (default: "text")
            pack_sequences: Whether to pack multiple sequences into one example
            pad_to_max_length: Whether to pad to max length
        """
        self.dataset_name = dataset_name
        self.subset = subset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.max_num_samples = max_num_samples
        self.seed = seed
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.streaming = streaming
        self.cache_dir = cache_dir
        self.num_proc = num_proc
        self.split = split
        self.text_field = text_field
        self.pack_sequences = pack_sequences
        self.pad_to_max_length = pad_to_max_length

        # Special tokens
        self.bos_id = tokenizer.bos_id if hasattr(tokenizer, 'bos_id') else tokenizer.tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_id if hasattr(tokenizer, 'eos_id') else tokenizer.tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_id if hasattr(tokenizer, 'pad_id') else tokenizer.tokenizer.pad_token_id

        # Load dataset
        self._load_dataset()

        # Buffer for packing sequences
        self.token_buffer = []
        self.current_index = 0

        logging.info(f"Loaded FineWeb dataset: {dataset_name}")
        if subset:
            logging.info(f"Using subset: {subset}")
        logging.info(f"Streaming mode: {streaming}")
        logging.info(f"Max sequence length: {max_seq_length}")
        if max_num_samples:
            logging.info(f"Limited to {max_num_samples} samples")

    def _load_dataset(self):
        """Load the FineWeb dataset from HuggingFace."""
        try:
            # Load dataset with optional subset
            if self.subset:
                self.hf_dataset = load_dataset(
                    self.dataset_name,
                    name=self.subset,
                    split=self.split,
                    streaming=self.streaming,
                    cache_dir=self.cache_dir,
                    num_proc=self.num_proc if not self.streaming else None,
                )
            else:
                self.hf_dataset = load_dataset(
                    self.dataset_name,
                    split=self.split,
                    streaming=self.streaming,
                    cache_dir=self.cache_dir,
                    num_proc=self.num_proc if not self.streaming else None,
                )

            # Shuffle if not streaming (streaming datasets are shuffled differently)
            if not self.streaming:
                self.hf_dataset = self.hf_dataset.shuffle(seed=self.seed)

            # Limit number of samples if specified
            if self.max_num_samples and not self.streaming:
                self.hf_dataset = self.hf_dataset.select(range(min(self.max_num_samples, len(self.hf_dataset))))

            # Create iterator for streaming datasets
            if self.streaming:
                self.dataset_iter = iter(self.hf_dataset)

            logging.info("Successfully loaded FineWeb dataset")

        except Exception as e:
            logging.error(f"Failed to load FineWeb dataset: {e}")
            raise

    def __len__(self):
        """Return dataset length."""
        if self.streaming:
            # For streaming datasets, we estimate length or use max_num_samples
            return self.max_num_samples if self.max_num_samples else 1000000  # Large number for streaming
        else:
            return len(self.hf_dataset)

    def __getitem__(self, idx):
        """Get a tokenized sample from the dataset."""
        if self.streaming:
            return self._get_streaming_sample()
        else:
            return self._get_indexed_sample(idx)

    def _get_streaming_sample(self):
        """Get a sample from streaming dataset."""
        try:
            # Get raw text
            raw_sample = next(self.dataset_iter)
            text = raw_sample[self.text_field]

            # Tokenize and process
            return self._process_text(text)

        except StopIteration:
            # Reset iterator if we reach the end
            self.dataset_iter = iter(self.hf_dataset)
            raw_sample = next(self.dataset_iter)
            text = raw_sample[self.text_field]
            return self._process_text(text)

    def _get_indexed_sample(self, idx):
        """Get a sample from indexed dataset."""
        raw_sample = self.hf_dataset[idx]
        text = raw_sample[self.text_field]
        return self._process_text(text)

    def _process_text(self, text):
        """Process and tokenize a text sample."""
        if not text or len(text.strip()) == 0:
            # Return empty sequence for invalid text
            return self._create_empty_sample()

        # Tokenize the text
        tokens = self.tokenizer.text_to_ids(text.strip())

        # Add special tokens
        if self.add_bos and self.bos_id is not None:
            tokens = [self.bos_id] + tokens
        if self.add_eos and self.eos_id is not None:
            tokens = tokens + [self.eos_id]

        # Handle sequence length
        if len(tokens) < self.min_seq_length:
            return self._create_empty_sample()

        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]

        # Convert to tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)

        # Create attention mask (all ones since no padding yet)
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)

        # Pad to max length if required
        if self.pad_to_max_length:
            padding_length = self.max_seq_length - len(input_ids)
            if padding_length > 0:
                pad_token = self.pad_id if self.pad_id is not None else 0
                input_ids = torch.cat([input_ids, torch.full((padding_length,), pad_token, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'length': torch.tensor(len(tokens), dtype=torch.long),
        }

    def _create_empty_sample(self):
        """Create an empty sample for invalid text."""
        if self.pad_to_max_length:
            pad_token = self.pad_id if self.pad_id is not None else 0
            input_ids = torch.full((self.max_seq_length,), pad_token, dtype=torch.long)
            attention_mask = torch.zeros(self.max_seq_length, dtype=torch.long)
        else:
            input_ids = torch.tensor([self.pad_id or 0], dtype=torch.long)
            attention_mask = torch.zeros(1, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'length': torch.tensor(0, dtype=torch.long),
        }

    def collate_fn(self, batch):
        """Collate function for batching samples."""
        # Get all input_ids and find max length in batch
        input_ids_list = [item['input_ids'] for item in batch]
        attention_mask_list = [item['attention_mask'] for item in batch]
        lengths = [item['length'] for item in batch]

        if not self.pad_to_max_length:
            # Dynamic padding to max length in batch
            max_len = max(len(ids) for ids in input_ids_list)
            pad_token = self.pad_id if self.pad_id is not None else 0

            padded_input_ids = []
            padded_attention_masks = []

            for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
                padding_length = max_len - len(input_ids)
                if padding_length > 0:
                    padded_input_ids.append(
                        torch.cat([input_ids, torch.full((padding_length,), pad_token, dtype=torch.long)])
                    )
                    padded_attention_masks.append(
                        torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
                    )
                else:
                    padded_input_ids.append(input_ids)
                    padded_attention_masks.append(attention_mask)

            input_ids_list = padded_input_ids
            attention_mask_list = padded_attention_masks

        # Stack tensors
        batch_input_ids = torch.stack(input_ids_list)
        batch_attention_mask = torch.stack(attention_mask_list)
        batch_lengths = torch.stack(lengths)

        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'lengths': batch_lengths,
        }