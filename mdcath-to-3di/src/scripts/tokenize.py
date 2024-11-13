import argparse

# python src/scripts/tokenize.py --input_path ./ --output_path ./ --tokenizers a, b, c

def main():
    parser = argparse.ArgumentParser(description="Tokenize input data")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')
    parser.add_argument('--tokenizers', type=str, nargs='+', required=True, help='List of tokenizers to use')

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    tokenizers = args.tokenizers

    print(f"Input Path: {input_path}")
    print(f"Output Path: {output_path}")
    print(f"Tokenizers: {tokenizers}")

if __name__ == "__main__":
    main()
