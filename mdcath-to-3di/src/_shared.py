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
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Increase shared memory limit
    os.environ["NCCL_SHM_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"

    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.max_rows", None)
    # torch.set_printoptions(profile="full")
    # torch.set_printoptions(profile="default")

    VERBOSE = train_config["verbose"]
    SEED = train_config["seed"]

    project_name = train_config["project_name"]
    custom_run_name = train_config["custom_run_name"]
    model_name_identifier = (
        project_name + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + (f"-{custom_run_name}" if custom_run_name else "")
    )

    USE_WANDB = train_config["weights_and_biases"]["enabled"]
    report_to = train_config["weights_and_biases"]["report_to"]

    if USE_WANDB:
        run = wandb.init(project=project_name, name=model_name_identifier)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    print("Model identifier:", model_name_identifier)

    return model_name_identifier, device, report_to, (run if USE_WANDB else None), USE_WANDB, SEED


def load_tokenizer(train_config):
    tokenizer = T5Tokenizer.from_pretrained(
        pretrained_model_name_or_path=train_config["model"]["protein_encoder_name"],
        do_lower_case=False,
        use_fast=True,
        legacy=False,
    )
    return tokenizer


def load_model(train_config, device):
    model = ProtT5(
        model_name=train_config["model"]["protein_encoder_name"],
        device=device
    )
    return model


def apply_lora_to_model():
    pass
