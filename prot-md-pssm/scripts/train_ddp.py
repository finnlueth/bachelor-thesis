import accelerate
import os
import random
import torch
import transformers
from accelerate import Accelerator

from src._shared import (
    load_config,
    setup_environment,
    load_tokenizer,
    prepare_dataset,
    load_model,
    apply_lora_to_model,
    setup_trainer,
    train_model,
    save_model_and_logs,
)
from src.utils.sanity_checks import sanity_checks


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["NCCL_SHM_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"

    train_config = load_config()
    model_name_identifier, device, report_to, run, USE_WANDB, SEED = setup_environment(train_config)

    accelerator = Accelerator()

    accelerate.utils.set_seed(SEED + 1)
    transformers.set_seed(SEED + 2)
    torch.manual_seed(SEED + 3)
    random.seed(SEED + 4)

    accelerator.wait_for_everyone()

    tokenizer = load_tokenizer(train_config)
    dataset = prepare_dataset(train_config)
    if accelerator.is_main_process:
        print(dataset)
        print(dataset[0])

    model = load_model(train_config, device)
    model = apply_lora_to_model(model, train_config)
    trainer = setup_trainer(train_config, tokenizer, model, model_name_identifier, USE_WANDB, dataset)

    accelerator.wait_for_everyone()

    model, trainer = accelerator.prepare(model, trainer)
    train_model(trainer)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        model_save_path = save_model_and_logs(unwrapped_model, train_config, model_name_identifier, trainer)
        sanity_checks(unwrapped_model, train_config, model_save_path)


if __name__ == "__main__":
    main()
