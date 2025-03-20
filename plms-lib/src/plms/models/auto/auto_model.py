from ..base_plm import ProteinLanguageModel
from ..T5.modeling_prostT5 import ProstT5
from ..T5.modeling_protT5 import ProtT5

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


def auto_model(model_name: str, *args, **kwargs) -> ProteinLanguageModel:
    if model_name not in MODEL_TYPES:
        raise ValueError(f"Model {model_name} not found in MODEL_TYPES")
    return MODEL_TYPES[model_name](name_or_path=model_name, *args, **kwargs)
