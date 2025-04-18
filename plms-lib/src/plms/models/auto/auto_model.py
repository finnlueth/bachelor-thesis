from ..base_plm import ProteinLanguageModel
from ...utils.constants import MODEL_TYPES


def auto_model(model_name: str, *args, **kwargs) -> ProteinLanguageModel:
    if model_name not in MODEL_TYPES:
        raise ValueError(f"Model {model_name} not found in MODEL_TYPES")
    return MODEL_TYPES[model_name](name_or_path=model_name, *args, **kwargs)
