#!/usr/bin/env python3
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

"""
Script to preprocess FineWeb dataset for Qwen3-Next training.

This script downloads FineWeb from HuggingFace and converts it to NeMo's
indexed binary format for efficient training.

Usage:
python scripts/nlp_language_modeling/preprocess_fineweb_for_qwen3_next.py \
    --output-prefix ./data/fineweb_qwen3 \
    --subset sample-10BT \
    --tokenizer-model Qwen/Qwen2-1.5B \
    --max-docs 100000 \
    --workers 8
"""

import argparse
import json
import multiprocessing
import os
import tempfile
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess FineWeb for Qwen3-Next training")

    # Dataset arguments
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="HuggingFaceFW/fineweb",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="sample-10BT",
        help="Dataset subset (e.g., sample-10BT, sample-350BT)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process"
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Name of the text field in the dataset"
    )

    # Output arguments
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Output file prefix for the processed dataset"
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Temporary directory for intermediate files"
    )

    # Tokenizer arguments
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default="Qwen/Qwen2-1.5B",
        help="HuggingFace tokenizer model name"
    )

    # Processing arguments
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to process (None for all)"
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=100,
        help="Minimum text length to include document"
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=100000,
        help="Maximum text length per document"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of worker processes"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing"
    )

    # Dataset format arguments
    parser.add_argument(
        "--dataset-impl",
        type=str,
        default="mmap",
        help="Dataset implementation (mmap or cached)"
    )
    parser.add_argument(
        "--append-eod",
        action="store_true",
        help="Append end-of-document token"
    )

    return parser.parse_args()


def create_temp_jsonl(dataset, text_field: str, max_docs: Optional[int], temp_file: str,
                     min_length: int, max_length: int):
    """Create temporary JSONL file from HuggingFace dataset."""
    print(f"Creating temporary JSONL file: {temp_file}")

    docs_written = 0
    docs_skipped = 0

    with open(temp_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(dataset, desc="Processing documents"):
            if max_docs is not None and docs_written >= max_docs:
                break

            text = sample[text_field]

            # Filter by length
            if len(text) < min_length:
                docs_skipped += 1
                continue

            # Truncate if too long
            if len(text) > max_length:
                text = text[:max_length]

            # Write as JSON line
            json_line = json.dumps({text_field: text}, ensure_ascii=False)
            f.write(json_line + '\n')
            docs_written += 1

            if docs_written % 10000 == 0:
                print(f"Processed {docs_written} documents, skipped {docs_skipped}")

    print(f"Finished creating JSONL file: {docs_written} documents, {docs_skipped} skipped")
    return docs_written


def main():
    """Main preprocessing function."""
    args = get_args()

    # Create output directory
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Set up temporary directory
    if args.temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    else:
        temp_dir = args.temp_dir
        os.makedirs(temp_dir, exist_ok=True)

    temp_jsonl = os.path.join(temp_dir, "fineweb_temp.jsonl")

    try:
        # Load dataset from HuggingFace
        print(f"Loading dataset: {args.dataset_name}")
        if args.subset:
            print(f"Using subset: {args.subset}")
            dataset = load_dataset(
                args.dataset_name,
                name=args.subset,
                split=args.split,
                streaming=True,  # Use streaming to handle large datasets
                trust_remote_code=True
            )
        else:
            dataset = load_dataset(
                args.dataset_name,
                split=args.split,
                streaming=True,
                trust_remote_code=True
            )

        # Create temporary JSONL file
        num_docs = create_temp_jsonl(
            dataset=dataset,
            text_field=args.text_field,
            max_docs=args.max_docs,
            temp_file=temp_jsonl,
            min_length=args.min_text_length,
            max_length=args.max_text_length
        )

        print(f"Created temporary JSONL with {num_docs} documents")

        # Initialize tokenizer
        print(f"Loading tokenizer: {args.tokenizer_model}")
        tokenizer = AutoTokenizer(args.tokenizer_model)

        # Build preprocessing command
        print("Starting NeMo preprocessing...")

        import subprocess

        preprocess_cmd = [
            "python",
            "scripts/nlp_language_modeling/preprocess_data_for_megatron.py",
            f"--input={temp_jsonl}",
            f"--json-keys={args.text_field}",
            f"--tokenizer-library=huggingface",
            f"--tokenizer-type={args.tokenizer_model}",
            f"--dataset-impl={args.dataset_impl}",
            f"--output-prefix={args.output_prefix}",
            f"--workers={args.workers}",
        ]

        if args.append_eod:
            preprocess_cmd.append("--append-eod")

        # Run preprocessing
        print(f"Running command: {' '.join(preprocess_cmd)}")
        result = subprocess.run(preprocess_cmd, check=True, capture_output=True, text=True)

        print("Preprocessing completed successfully!")
        print("Output files created:")
        print(f"  {args.output_prefix}_text_document.bin")
        print(f"  {args.output_prefix}_text_document.idx")

        # Print stats
        bin_file = f"{args.output_prefix}_text_document.bin"
        if os.path.exists(bin_file):
            size_mb = os.path.getsize(bin_file) / (1024 * 1024)
            print(f"Binary file size: {size_mb:.1f} MB")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise

    finally:
        # Cleanup temporary files
        if os.path.exists(temp_jsonl):
            os.remove(temp_jsonl)
            print(f"Cleaned up temporary file: {temp_jsonl}")

        if args.temp_dir is None:
            os.rmdir(temp_dir)


if __name__ == "__main__":
    main()