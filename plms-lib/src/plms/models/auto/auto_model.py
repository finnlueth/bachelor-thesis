import torch
from transformers import PreTrainedModel

from ...configurations.configuration_base_plm import PLMConfig
from ..T5.prostT5 import ProstT5
from ..T5.protT5 import ProtT5

MODEL_TYPES = {
    # "ElnaggarLab/ankh-base",
    # "ElnaggarLab/ankh-large",
    # "facebook/esm2_t6_8M_UR50D",
    # "facebook/esm2_t12_35M_UR50D",
    # "facebook/esm2_t30_150M_UR50D",
    # "facebook/esm2_t33_650M_UR50D",
    # "facebook/esm2_t36_3B_UR50D",
    "Rostlab/prot_t5_xl_half_uniref50-enc": ProtT5,
    "Rostlab/prot_t5_xl_uniref50": ProtT5,
    "Rostlab/ProstT5_fp16": ProstT5,
    "Rostlab/ProstT5": ProstT5,
}


def auto_model(model_name: str, *args, **kwargs) -> PreTrainedModel:
    if model_name not in MODEL_TYPES:
        raise ValueError(f"Model {model_name} not found in MODEL_TYPES")

    config = MODEL_TYPES[model_name].get_default_config()
    return MODEL_TYPES[model_name](config=config, *args, **kwargs)
