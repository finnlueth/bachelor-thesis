from typing import List, Union

import re
import torch
from transformers import T5Tokenizer

from ..base_tokenizer import ProteinLanguageModelTokenizer


class ProtT5Tokenizer(ProteinLanguageModelTokenizer):
    def __init__(self, name_or_path, tokenizer_class=T5Tokenizer, *args, **kwargs):
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
    ) -> List[str]:
        if isinstance(text, str):
            text = [text]
        if isinstance(text, list):
            text = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in text]
        else:
            raise TypeError("text must be a string or a list of strings")

        return self(
            text=text,
            padding=padding,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
            *args,
            **kwargs,
        )

    def decode(
        self,
        tokens: Union[torch.Tensor, List[int], List[List[int]]],
        skip_special_tokens=True,
        *args,
        **kwargs,
    ) -> Union[str, List[str]]:
        return [
            seq.replace(" ", "")
            for seq in self.tokenizer.batch_decode(tokens, skip_special_tokens=skip_special_tokens, *args, **kwargs)
        ]
