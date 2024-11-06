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
import yaml

from src.model.model import T5EncoderModelForPssmGeneration, compute_metrics
from src.model.utils import DataCollatorForT5Pssm

with open('../configs/train.yml', 'r') as file:
    CONFIG = yaml.safe_load(file)
with open('../configs/project.yml', 'r') as file:
    FILE_PATHS = yaml.safe_load(file)['paths']

device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

MODEL_VERRSION = CONFIG['model_version']
SEED = CONFIG['seed']
BASE_MODEL = CONFIG['base_model']

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
set_seed(SEED)

TRAINING_CONFIG = CONFIG['training_config']
