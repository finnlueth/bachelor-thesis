from transformers import T5Tokenizer




def get_tokenizer(name_or_path) -> T5Tokenizer:
    tokenizer = T5Tokenizer.from_pretrained(
        pretrained_model_name_or_path=name_or_path,
        do_lower_case=False,
        use_fast=True,
        legacy=False,
    )
    return tokenizer