import argparse
import importlib
import logging
import os
import concurrent.futures
from src.utils.logging import setup_logging
from src.data.tokenize import (
    FoldSeekTokenizer,
    FoldToken4Tokenizer,
    Bio2TokenTokenizer
)
from src.data.load import (
    MDCATHDataset,
    MisatoDataset,
    AtlasDataset
)
# from src.data.utils import save_to_h5

# python src/scripts/tokenize.py --input_path ./ --output_path ./

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Tokenize input data")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input file/directory. If directory, all files in the directory will be processed. If file, only the file will be processed.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--tokenizers', type=str, nargs='+', choices=['bio2token', 'foldseek', 'foldtoken4'], default=['bio2token', 'foldseek', 'foldtoken4'], help='List of tokenizers to use')
    parser.add_argument('--dataset', type=str, required=True, choices=['mdcath', 'misato', 'atlas'], help='Type of dataset structure to process')

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    tokenizers = args.tokenizers
    dataset = args.dataset

    logger.info("Input Path: %s", input_path)
    logger.info("Output Path: %s", output_path)
    logger.info("Tokenizers: %s", tokenizers)

    tokenizer_classes = {
        'bio2token': Bio2TokenTokenizer,
        'foldseek': FoldSeekTokenizer,
        'foldtoken4': FoldToken4Tokenizer
    }
    tokenizer_classes = [tokenizer_classes[t] for t in tokenizers]
    
    dataset_class = {
        'mdcath': MDCATHDataset,
        'misato': MisatoDataset,
        'atlas': AtlasDataset
    }
    dataset_class = dataset_class[dataset]
    
    

    def process_file(file_path):
        with open(file_path, 'r') as f:
            pdb_content = f.read()
        
        pdb_files = [pdb_content]  # Assuming each file contains one PDB structure

        for tokenizer_name in tokenizers:
            tokenizer_class = tokenizer_classes.get(tokenizer_name)
            if tokenizer_class:
                tokenizer = tokenizer_class()
                logger.info("Running %s tokenizer on %s...", tokenizer_name, file_path)
                tokenized_data = tokenizer.tokenize(pdb_files)
                output_file_path = os.path.join(output_path, f"{tokenizer_name}_tokenized.h5")
                save_to_h5(tokenized_data, output_file_path)
            else:
                logger.warning("Tokenizer %s is not implemented.", tokenizer_name)

    process_dir = args.process_dir

    if process_dir:
        if os.path.isdir(input_path):
            files = [os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
                executor.map(process_file, files)
        else:
            logger.error("Input path is not a directory.")
    else:
        process_file(input_path)

if __name__ == "__main__":
    main()
