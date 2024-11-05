import copy
import gc
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peft
import torch
import torch.nn as nn
import tqdm
from datasets import Dataset, DatasetDict
from peft import (
    LoraConfig,
)
from transformers import (
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    T5Config,
    T5ForTokenClassification,
    T5Tokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from src.model.model import T5EncoderModelForPssmGeneration, compute_metrics
from src.model.utils import DataCollatorForT5Pssm

device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

MODEL_VERRSION = 0.1
SEED = 69420
BASE_MODEL = "Rostlab/prot_t5_xl_uniref50"
# BASE_MODEL = "facebook/esm2_t6_8M_UR50D"
VERBOSE = True
FILE_PATHS = {
    "pssm": "../tmp/data/pssm",
    "models": "../tmp/models/",
}

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
set_seed(SEED)

TRAINING_CONFIG = {
    'learning_rate': 1e-4,
    'batch_size': 2,
    'num_epochs': 10,
    'logging_steps': 1,
    'eval_steps': 300,
    'save_steps': 9999999,
}