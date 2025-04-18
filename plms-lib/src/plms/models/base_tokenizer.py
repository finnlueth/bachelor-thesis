from abc import abstractmethod
from typing import Dict, List, Union
import os

import torch


class ProteinLanguageModelTokenizer:
    """Tokenizer for protein language models."""

    def __init__(self, tokenizer_class, *args, **kwargs):
        self.tokenizer = tokenizer_class.from_pretrained(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Call the tokenizer."""
        return self.tokenizer(*args, **kwargs)

    def tokenize(self, text: Union[str, List[str]], *args, **kwargs) -> Union[List[int], List[List[int]]]:
        """Tokenize a text."""
        return self.encode(text, *args, **kwargs)

    @abstractmethod
    def encode(self, text: Union[str, List[str]], *args, **kwargs) -> Union[List[int], List[List[int]]]:
        """Encode a text."""
        pass

    @abstractmethod
    def decode(self, tokens: Union[torch.Tensor, List[int]], *args, **kwargs) -> Union[str, List[str]]:
        """Decode a list of tokens into a string.
        If a list of tokenized sequences is provided, the function will return a list of strings.
        If a single tokenized sequence is provided, the function will return a string.
        """
        pass
    
    def get_tokenizer(self):
        return self.tokenizer

    def tokenize_fasta(
        self,
        fasta: Union[str, os.PathLike],
        return_headers: bool = False,
        *args,
        **kwargs,
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Tokenize a fasta file."""
        if isinstance(fasta, str):
            if "\n" not in fasta or type(fasta) is os.PathLike:
                with open(fasta, "r") as f:
                    sequences = f.read()
            else:
                sequences = fasta
        else:
            raise TypeError("fasta must be a a fasta formatted string or a file path")

        sequences = [(seq.strip().split("\n")[0], "".join(seq.strip().split("\n")[1:])) for seq in sequences.split(">")[1:]]
        headers, protein_seqs = zip(*sequences)
        headers = list(headers)
        protein_seqs = list(protein_seqs)

        tokenized_seqs = self.tokenize(protein_seqs, *args, **kwargs)

        if return_headers:
            tokenized_seqs["headers"] = headers
        return tokenized_seqs
