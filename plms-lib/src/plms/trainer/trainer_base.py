from transformers import Trainer
from torch.utils.data import DataLoader
from collections import defaultdict
import random

from transformers.trainer_utils import (
    seed_worker,
)

from transformers.utils import (
    is_datasets_available,
)

import datasets
import torch


class ProteinLanguageModelTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)