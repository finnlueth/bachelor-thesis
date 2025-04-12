import gc
import os
import random
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from datasets import load_from_disk
from peft import (
    LoraConfig,
    PeftConfig,
    get_peft_model,
)
from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from plms.models import ProtT5
from transformers import (
    AutoConfig,
    AutoTokenizer,
    T5Tokenizer,
    TrainingArguments,
)

import src.model.utils as utils
from src.model.configuration_md_pssm import MDPSSMConfig
from src.model.metrics import compute_metrics
from src.model.modeling_md_pssm import T5EncoderModelForPssmGeneration
from src.model.trainer_protein_subset import ProteinSampleSubsetTrainer
from src.model.utils.data_collator import DataCollatorForT5Pssm
from src.plots.train_plots import plot_training_history

# from accelerate.distributed import DistributedDataParallelKwargs
# from src.model.configuration_protein_clip import ProtT5CLIPConfig
# from src.model.data_collator_multi_input import DataCollatorForProtT5CLIP
# from src.model.metrics import metrics_factory
# from src.model.modeling_protein_clip import ProtT5CLIP
# from src.model.trainer_protein_subset import ProteinSampleSubsetTrainer


def load_config():
    with open("../configs/model.yaml", "r") as f:
        train_config = yaml.safe_load(f)
    return train_config


def setup_environment(train_config):
    """Setup training environment and configs"""
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Increase shared memory limit
    # os.environ["NCCL_SHM_DISABLE"] = "1"
    # os.environ["NCCL_P2P_DISABLE"] = "1"

    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.max_rows", None)
    # torch.set_printoptions(profile="full")
    # torch.set_printoptions(profile="default")



    project_name = train_config["project_name"]
    custom_run_name = train_config["custom_run_name"].replace(" ", "-")
    model_name_identifier = (
        project_name + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + (f"-{custom_run_name.replace(' ', '-')}" if custom_run_name else "")
    )

    USE_WANDB = train_config["weights_and_biases"]["enabled"]
    report_to = train_config["weights_and_biases"]["report_to"]

    if USE_WANDB:
        run = wandb.init(project=project_name, name=model_name_identifier)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    print("Model identifier:", model_name_identifier)

    return model_name_identifier, device, report_to, (run if USE_WANDB else None), USE_WANDB, SEED

def load_model(train_config, device):
    config = MDPSSMConfig(device=device)
    model = T5EncoderModelForPssmGeneration(config)
    model.to(device)
    print(model)
    return model

def apply_lora_to_model(model, train_config):
    target_modules = ["q", "v"]
    modules_to_save = ["pssm_head"]

    lora_config = LoraConfig(
        inference_mode=False,
        r=train_config["lora"]["r"],
        lora_alpha=train_config["lora"]["lora_alpha"],
        lora_dropout=train_config["lora"]["lora_dropout"],
        target_modules=target_modules,
        bias="none",
        modules_to_save=modules_to_save,
        use_rslora=train_config["lora"]["use_rslora"],
        use_dora=train_config["lora"]["use_dora"],
    )

    model = get_peft_model(model, lora_config)

    print("target_modules:", target_modules)
    print("modules_to_save:", modules_to_save)
    model.print_trainable_parameters()

    return model


def load_tokenizer(train_config):
    tokenizer = T5Tokenizer.from_pretrained(
        pretrained_model_name_or_path=train_config["model"]["protein_encoder_name"],
        do_lower_case=False,
        use_fast=True,
        legacy=False,
    )
    return tokenizer


def prepare_dataset(train_config):
    dataset = load_from_disk(train_config["dataset"]["path"])
    # dataset = dataset.select(range(100))
    # print(dataset)

    # for x in range(0, 18, 4):
    #     print(dataset[x]["name"])
    #     print(dataset[x]["sequence"])
    #     print(torch.tensor(dataset[x]["pssm_features"]).shape)
    # display(torch.tensor(dataset[x]["pssm_features"]))

    dataset = dataset.rename_column("pssm_features", "labels")
    dataset = dataset.remove_columns(["name", "sequence", "sequence_processed"])

    # print(dataset)

    return dataset


def setup_trainer(train_config, tokenizer, model, model_name_identifier, USE_WANDB, dataset):
    data_collator = DataCollatorForT5Pssm(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=f"../tmp/models/checkpoints/{model_name_identifier}",
        run_name=model_name_identifier if USE_WANDB else None,
        report_to="wandb" if USE_WANDB else None,
        learning_rate=train_config["trainer"]["learning_rate"],
        per_device_train_batch_size=train_config["trainer"]["train_batch_size"],
        num_train_epochs=train_config["trainer"]["num_epochs"],
        eval_strategy=train_config["trainer"]["eval_strategy"],
        eval_steps=train_config["trainer"]["eval_steps"],
        per_device_eval_batch_size=train_config["trainer"]["eval_batch_size"],
        eval_on_start=train_config["trainer"]["eval_on_start"],
        batch_eval_metrics=train_config["trainer"]["batch_eval_metrics"],
        save_strategy=train_config["trainer"]["save_strategy"],
        save_steps=train_config["trainer"]["save_steps"],
        save_total_limit=train_config["trainer"]["save_total_limit"],
        remove_unused_columns=train_config["trainer"]["remove_unused_columns"],
        label_names=["input_ids", "attention_mask"],
        logging_strategy="steps",
        logging_steps=train_config["trainer"]["logging_steps"],
        seed=train_config["seed"],
        lr_scheduler_type=train_config["trainer"]["lr_scheduler_type"],
        warmup_steps=train_config["trainer"]["warmup_steps"],
    )

    trainer = ProteinSampleSubsetTrainer(
    # trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # eval_sample_size=train_config["trainer"]["eval_sample_size"],
    )
    return trainer


def train_model(trainer):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    trainer.train()
    trainer.evaluate()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def save_model_and_logs(model, train_config, model_name_identifier, trainer):
    model_save_path = f"../tmp/models/adapters/{model_name_identifier}"

    model.save_pretrained(save_directory=model_save_path)

    pd.DataFrame(trainer.state.log_history).to_csv(f"{model_save_path}/training_log.csv", index=False)

    with open(f"{model_save_path}/train_config.yaml", "w") as f:
        train_config["model"]["reload_from_checkpoint_path"] = model_save_path
        yaml.dump(train_config, f, sort_keys=False)

    fig = plot_training_history(
        log_history=pd.DataFrame(trainer.state.log_history), train_config=train_config, metrics_names=["loss"]
    )
    fig.savefig(f"{model_save_path}/training_history.png")
    plt.close(fig)

    print("Model, config, and log saved to:", model_save_path)
    
    return model_save_path
