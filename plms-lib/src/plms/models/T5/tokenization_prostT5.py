from typing import List, Union

import re
import torch
from transformers import T5Tokenizer

from ..base_tokenizer import ProteinLanguageModelTokenizer


class ProstT5Tokenizer(ProteinLanguageModelTokenizer):
    def __init__(self, name_or_path="Rostlab/ProstT5", tokenizer_class=T5Tokenizer, *args, **kwargs):
        super().__init__(
            tokenizer_class=tokenizer_class,
            pretrained_model_name_or_path=name_or_path,
            do_lower_case=False,
            use_fast=True,
            legacy=False,
            *args,
            **kwargs,
        )

    def encode(
        self,
        text: Union[str, List[str]],
        padding: bool = False,
        truncation: bool = False,
        add_special_tokens: bool = True,
        *args,
        **kwargs,
    ) -> List[int]:
        def preprocess_text(text: str) -> str:
            return (
                "<AA2fold> " + " ".join(list(re.sub(r"[UZOB]", "X", text)))
                if text.isupper()
                else "<fold2AA> " + " ".join(list(re.sub(r"[uzob]", "X", text)))
            )

        if isinstance(text, str):
            text = preprocess_text(text)
        elif isinstance(text, list):
            text = [preprocess_text(sequence) for sequence in text]
        else:
            raise TypeError("text must be a string or a list of strings")

        encoded = self(
            text=text,
            padding=padding,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
            *args,
            **kwargs,
        )

        if len(encoded) == 1:
            encoded = encoded[0]

        return encoded

    def decode(
        self,
        tokens: Union[torch.Tensor, List[int], List[List[int]]],
        skip_special_tokens=True,
        *args,
        **kwargs,
    ) -> Union[str, List[str]]:
        if isinstance(tokens, torch.Tensor) and tokens.shape[0] == 1:
            tokens = tokens.squeeze(0)

        if isinstance(tokens, list) and all(isinstance(x, int) for x in tokens):
            tokens = [tokens]

        def postprocess_text(text: str) -> str:
            return (
                text.replace(" ", "").replace("<AA2fold>", "").replace("<fold2AA>", "")
                if skip_special_tokens
                else text.replace(" ", "")
            )

        decoded = [
            postprocess_text(seq)
            for seq in self.tokenizer.batch_decode(tokens, skip_special_tokens=skip_special_tokens, *args, **kwargs)
        ]

        if len(decoded) == 1:
            return decoded[0]
        else:
            return decoded
