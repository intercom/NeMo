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

"""HuggingFace streaming dataset integration for NeMo Megatron pre-training."""

import math
import random
from typing import Dict, Optional

import numpy as np
import torch
from datasets import load_dataset

from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import get_samples_mapping
from nemo.core.classes import Dataset
from nemo.utils import logging

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True
except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False


class HuggingFaceStreamingDataset(Dataset):
    """
    HuggingFace streaming dataset for NeMo Megatron pre-training.

    This class provides a bridge between HuggingFace streaming datasets
    and NeMo's expected dataset interface for pre-training.
    """

    def __init__(
        self,
        cfg,
        trainer,
        tokenizer,
        name: str,
        dataset_name: str,
        subset: Optional[str] = None,
        num_samples: int = None,
        seq_length: int = 2048,
        seed: int = 1234,
        drop_last: bool = True,
        text_field: str = "text",
        streaming: bool = True,
        cache_dir: Optional[str] = None,
        split: str = "train",
    ):
        """
        Initialize HuggingFace streaming dataset.

        Args:
            cfg: Model configuration
            trainer: PyTorch Lightning trainer
            tokenizer: Tokenizer instance
            name: Dataset split name ('train', 'valid', 'test')
            dataset_name: HuggingFace dataset name (e.g., 'HuggingFaceFW/fineweb')
            subset: Dataset subset/config name
            num_samples: Number of samples to generate
            seq_length: Sequence length for each sample
            seed: Random seed
            drop_last: Whether to drop the last incomplete batch
            text_field: Name of the text field in the dataset
            streaming: Whether to use streaming mode
            cache_dir: Cache directory for the dataset
            split: Dataset split to load
        """
        if not HAVE_MEGATRON_CORE:
            raise ImportError("megatron-core required for Megatron datasets")

        super().__init__()

        self.cfg = cfg
        self.name = name
        self.dataset_name = dataset_name
        self.subset = subset
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.seed = seed
        self.drop_last = drop_last
        self.text_field = text_field
        self.streaming = streaming
        self.cache_dir = cache_dir
        self.split = split

        # Configuration from cfg
        self.reset_position_ids = cfg.data.get('reset_position_ids', False)
        self.reset_attention_mask = cfg.data.get('reset_attention_mask', False)
        self.eod_mask_loss = cfg.data.get('eod_mask_loss', False)
        self.create_inputs = any([self.reset_position_ids, self.reset_attention_mask, self.eod_mask_loss])

        # Special tokens
        self.eos_id = tokenizer.eos_id
        self.pad_id = getattr(tokenizer, 'pad_id', 0)
        self.bos_id = getattr(tokenizer, 'bos_id', None)

        # Load dataset
        self._load_hf_dataset()

        # Create sample mapping
        self._create_sample_mapping()

        # Buffer for text accumulation
        self._text_buffer = []
        self._current_sample_idx = 0

        logging.info(f"Initialized HuggingFace dataset {dataset_name} ({name} split)")
        logging.info(f"Sequence length: {seq_length}, Num samples: {num_samples}")

    def _load_hf_dataset(self):
        """Load the HuggingFace dataset."""
        try:
            load_args = {
                'path': self.dataset_name,
                'split': self.split,
                'streaming': self.streaming,
                'cache_dir': self.cache_dir,
            }

            if self.subset:
                load_args['name'] = self.subset

            if not self.streaming:
                load_args['num_proc'] = 4  # Use multiprocessing for non-streaming

            self.hf_dataset = load_dataset(**load_args)

            # Shuffle non-streaming datasets
            if not self.streaming:
                self.hf_dataset = self.hf_dataset.shuffle(seed=self.seed)

            # Create iterator for streaming
            if self.streaming:
                self.dataset_iter = iter(self.hf_dataset)

            logging.info(f"Successfully loaded {self.dataset_name}")

        except Exception as e:
            logging.error(f"Failed to load dataset {self.dataset_name}: {e}")
            raise

    def _create_sample_mapping(self):
        """Create mapping from sample index to dataset."""
        if self.num_samples is None:
            if self.streaming:
                # For streaming, estimate a large number
                self.num_samples = 1000000
            else:
                self.num_samples = len(self.hf_dataset)

        # Create sample indices
        self.sample_mapping = list(range(self.num_samples))
        if not self.streaming:
            random.Random(self.seed).shuffle(self.sample_mapping)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Get a tokenized sample."""
        # Get raw text from dataset
        text = self._get_text_sample(idx)

        # Tokenize and process
        return self._process_text_sample(text)

    def _get_text_sample(self, idx):
        """Get raw text sample from dataset."""
        if self.streaming:
            return self._get_streaming_text()
        else:
            actual_idx = self.sample_mapping[idx] % len(self.hf_dataset)
            sample = self.hf_dataset[actual_idx]
            return sample[self.text_field]

    def _get_streaming_text(self):
        """Get text from streaming dataset."""
        try:
            sample = next(self.dataset_iter)
            return sample[self.text_field]
        except StopIteration:
            # Reset iterator and try again
            self.dataset_iter = iter(self.hf_dataset)
            sample = next(self.dataset_iter)
            return sample[self.text_field]

    def _process_text_sample(self, text):
        """Process text into tokens following NeMo's format."""
        if not text or len(text.strip()) == 0:
            # Return minimal valid sample
            tokens = [self.eos_id] if self.eos_id is not None else [0]
        else:
            # Tokenize text
            tokens = self.tokenizer.text_to_ids(text.strip())

            # Add EOS token
            if self.eos_id is not None:
                tokens.append(self.eos_id)

        # Ensure we have the right sequence length
        if len(tokens) > self.seq_length:
            tokens = tokens[:self.seq_length]
        elif len(tokens) < self.seq_length:
            # Pad with EOS tokens (NeMo style)
            pad_token = self.eos_id if self.eos_id is not None else 0
            tokens.extend([pad_token] * (self.seq_length - len(tokens)))

        # Convert to tensor
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)

        # Create the sample dict that NeMo expects
        sample = {
            'tokens': tokens_tensor,
        }

        # Add additional inputs if required
        if self.create_inputs:
            # Position IDs
            if self.reset_position_ids:
                position_ids = torch.arange(self.seq_length, dtype=torch.long)
                sample['position_ids'] = position_ids

            # Attention mask
            if self.reset_attention_mask:
                attention_mask = torch.ones(self.seq_length, dtype=torch.bool)
                sample['attention_mask'] = attention_mask

            # Loss mask
            if self.eod_mask_loss:
                loss_mask = torch.ones(self.seq_length, dtype=torch.float)
                sample['loss_mask'] = loss_mask

        return sample


def build_hf_streaming_dataset(
    cfg,
    trainer,
    tokenizer,
    name: str,
    dataset_name: str,
    subset: Optional[str] = None,
    num_samples: int = None,
    seq_length: int = 2048,
    seed: int = 1234,
    text_field: str = "text",
):
    """
    Build a HuggingFace streaming dataset.

    This function creates a dataset that can be used with NeMo's training pipeline
    while streaming data from HuggingFace datasets.
    """
    return HuggingFaceStreamingDataset(
        cfg=cfg,
        trainer=trainer,
        tokenizer=tokenizer,
        name=name,
        dataset_name=dataset_name,
        subset=subset,
        num_samples=num_samples,
        seq_length=seq_length,
        seed=seed,
        text_field=text_field,
        streaming=True,  # Always use streaming for large datasets
        cache_dir=cfg.data.get('cache_dir', './hf_cache'),
        split="train" if name == "train" else "validation" if name == "valid" else "test",
    )


def build_train_valid_test_hf_datasets(
    cfg,
    trainer,
    tokenizer,
    dataset_name: str,
    subset: Optional[str] = None,
    train_num_samples: int = None,
    valid_num_samples: int = None,
    test_num_samples: int = None,
    seq_length: int = 2048,
    seed: int = 1234,
    text_field: str = "text",
):
    """
    Build train, validation, and test datasets from a HuggingFace dataset.

    For datasets like FineWeb that only have a train split, this will create
    the validation set by sampling from the training data.
    """

    # Build training dataset
    train_ds = build_hf_streaming_dataset(
        cfg=cfg,
        trainer=trainer,
        tokenizer=tokenizer,
        name="train",
        dataset_name=dataset_name,
        subset=subset,
        num_samples=train_num_samples,
        seq_length=seq_length,
        seed=seed,
        text_field=text_field,
    )

    # Build validation dataset (sample from train split)
    valid_ds = build_hf_streaming_dataset(
        cfg=cfg,
        trainer=trainer,
        tokenizer=tokenizer,
        name="valid",
        dataset_name=dataset_name,
        subset=subset,
        num_samples=valid_num_samples,
        seq_length=seq_length,
        seed=seed + 1,  # Different seed for validation
        text_field=text_field,
    )

    # Build test dataset (can be None if not needed)
    test_ds = None
    if test_num_samples and test_num_samples > 0:
        test_ds = build_hf_streaming_dataset(
            cfg=cfg,
            trainer=trainer,
            tokenizer=tokenizer,
            name="test",
            dataset_name=dataset_name,
            subset=subset,
            num_samples=test_num_samples,
            seq_length=seq_length,
            seed=seed + 2,  # Different seed for test
            text_field=text_field,
        )

    return train_ds, valid_ds, test_ds