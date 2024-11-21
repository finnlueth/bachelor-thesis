import argparse
import importlib
import logging
from src.utils.logging import setup_logging

# python src/scripts/tokenize.py --input_path ./ --output_path ./

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Tokenize input data")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')
    parser.add_argument('--tokenizers', type=str, nargs='+', choices=['bio2token', 'foldseek', 'foldtoken'], default=['bio2token', 'foldseek', 'foldtoken'], help='List of tokenizers to use')

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    tokenizers = args.tokenizers

    logger.info("Input Path: %s", input_path)
    logger.info("Output Path: %s", output_path)
    logger.info("Tokenizers: %s", tokenizers)

    for tokenizer in tokenizers:
        try:
            module = importlib.import_module(f'src.data.{tokenizer}')
            if hasattr(module, 'tokenize'):
                logger.info("Running %s tokenizer...", tokenizer)
                module.tokenize(input_path, output_path)
            else:
                logger.warning("Tokenizer %s does not have a 'tokenize' function.", tokenizer)
        except ImportError:
            logger.error("Tokenizer %s module not found in src.data.", tokenizer)

if __name__ == "__main__":
    main()
