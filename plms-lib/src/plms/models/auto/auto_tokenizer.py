from ..base_tokenizer import ProteinLanguageModelTokenizer
from ...utils.constants import TOKENIZER_TYPES


def auto_tokenizer(model_name: str, *args, **kwargs) -> ProteinLanguageModelTokenizer:
    if model_name not in TOKENIZER_TYPES:
        raise ValueError(f"Model {model_name} not found in TOKENIZER_TYPES")
    return TOKENIZER_TYPES[model_name](name_or_path=model_name, *args, **kwargs)
