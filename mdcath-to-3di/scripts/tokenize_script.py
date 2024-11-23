#! /usr/bin/env python3
"""
python3 scripts/tokenize_script.py --input_path ./tmp/data/mdCATH/data/ --output_path ./tmp/output/tokenized/mdcath --tokenizers bio2token foldseek --dataset mdcath
"""

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from src.data.load import (
    AtlasDataset,
    MDCATHDataset,
    MisatoDataset,
    TrajectoryDataset,
)
from src.data.tokenize import (
    Bio2TokenTokenizer,
    FoldSeekTokenizer,
    FoldToken4Tokenizer,
)
from src.utils.logging import setup_logging


def process_dataset_parallel(dataset: TrajectoryDataset, tokenizer_wrappers: dict, max_workers: int = os.cpu_count() - 1):
    def _process_item(idx: int):
        try:
            item = dataset[idx]
            results = {}
            for tokenizer_name, tokenizer_wrapper in tokenizer_wrappers.items():
                results[tokenizer_name] = "123"  # tokenizer_wrapper.tokenize(item)
            dataset.use_trajectory_location(idx)
            return dataset[idx]
        except ValueError as e:
            logging.error("Error processing item %s: %s", idx, e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_process_item, range(len(dataset))))

    return results


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Tokenize input trajectory files")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input file/directory. If directory, all files in\
            the directory will be processed. If file, only the file will be processed.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--tokenizers",
        type=str,
        nargs="+",
        choices=["bio2token", "foldseek", "foldtoken4"],
        default=["bio2token", "foldseek", "foldtoken4"],
        help="List of tokenizers to use.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mdcath", "misato", "atlas"],
        help="Type of dataset structure to process.",
    )

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    tokenizers = args.tokenizers
    dataset = args.dataset

    logging.info("Input Path: %s", input_path)
    logging.info("Output Path: %s", output_path)
    logging.info("Tokenizers: %s", tokenizers)

    tokenizer_wrappers = {"foldseek": FoldSeekTokenizer, "bio2token": Bio2TokenTokenizer, "foldtoken4": FoldToken4Tokenizer}
    tokenizer_wrappers = {t: tokenizer_wrappers[t]() for t in tokenizers}

    dataset_wrapper = {"mdcath": MDCATHDataset, "misato": MisatoDataset, "atlas": AtlasDataset}
    dataset_wrapper = dataset_wrapper[dataset](data_dir=input_path)

    results = process_dataset_parallel(dataset_wrapper, tokenizer_wrappers)
    logging.info("Results: %s", results)


if __name__ == "__main__":
    main()
