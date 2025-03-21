from ..base_tokenizer import ProteinLanguageModelTokenizer
from ..T5.tokenization_protT5 import ProtT5Tokenizer
from ..T5.tokenization_prostT5 import ProstT5Tokenizer

TOKENIZER_TYPES = {
    "Rostlab/prot_t5_xl_half_uniref50-enc": ProtT5Tokenizer,
    "Rostlab/prot_t5_xl_uniref50": ProtT5Tokenizer,
    "Rostlab/ProstT5_fp16": ProtT5Tokenizer,
    "Rostlab/ProstT5": ProstT5Tokenizer,
}


def auto_tokenizer(model_name: str, *args, **kwargs) -> ProteinLanguageModelTokenizer:
    if model_name not in TOKENIZER_TYPES:
        raise ValueError(f"Model {model_name} not found in TOKENIZER_TYPES")
    return TOKENIZER_TYPES[model_name](name_or_path=model_name, *args, **kwargs)
