from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from transformers.data.data_collator import DataCollatorForTokenClassification

import pandas as pd
from IPython.display import display

import torch


@dataclass
class DataCollatorForT5Pssm:
    """
    Adapted from transformers.data.data_collator.DataCollatorForTokenClassification
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        label_name = "label" if "label" in features[0].keys() else "labels"

        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        no_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]
        print("no_labels_features", no_labels_features)

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                [to_list(row) + [self.label_pad_token_id] * (sequence_length - len(row) - 1) for row in label] for label in labels
            ]
        else:
            raise ValueError("padding_side must be 'right'")

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.float32)

        return batch
