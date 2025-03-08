import os
import random

import accelerate
import torch
import transformers

from src._shared import (
    apply_lora_to_model,
    load_config,
    load_model,
    load_tokenizer,
    prepare_dataset,
    save_model_and_logs,
    setup_environment,
    setup_trainer,
    train_model,
)
from src.utils.sanity_checks import sanity_checks


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train_config = load_config()
    model_name_identifier, device, report_to, run, USE_WANDB, SEED = setup_environment(train_config)

    accelerate.utils.set_seed(SEED + 1)
    transformers.set_seed(SEED + 2)
    torch.manual_seed(SEED + 3)
    random.seed(SEED + 4)

    tokenizer = load_tokenizer(train_config)
    dataset = prepare_dataset(train_config)
    print(dataset)
    print(dataset[0])

    model = load_model(train_config, device)
    model = apply_lora_to_model(model, train_config)
    
    trainer = setup_trainer(train_config, tokenizer, model, model_name_identifier, USE_WANDB, dataset)
    train_model(trainer)
    
    model_save_path = save_model_and_logs(model, train_config, model_name_identifier, trainer)
    sanity_checks(model, train_config, model_save_path)


if __name__ == "__main__":
    main()
