from ..models import ProstT5, ProtT5, ProtT5Tokenizer, ProstT5Tokenizer

ALPHABET_AA = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
ALPHABET_PROST = ["a", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "y"]

MODEL_TYPES = {
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

TOKENIZER_TYPES = {
    "Rostlab/prot_t5_xl_half_uniref50-enc": ProtT5Tokenizer,
    "Rostlab/prot_t5_xl_uniref50": ProtT5Tokenizer,
    "Rostlab/ProstT5_fp16": ProstT5Tokenizer,
    "Rostlab/ProstT5": ProstT5Tokenizer,
}