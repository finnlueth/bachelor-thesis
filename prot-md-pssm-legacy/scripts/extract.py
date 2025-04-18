import gc
import os
from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pandas as pd
import torch
import wandb
import yaml
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from plms import PLMConfig, ProstT5, ProstT5Tokenizer
from torch import nn
from torch.nn import (
    KLDivLoss,
)
from transformers import (
    PreTrainedModel,
    TrainingArguments,
)

from src.model.configuration_md_pssm import MDPSSMConfig
from src.model.metrics import compute_metrics
from src.model.modeling_outputs import PSSMOutput
from src.model.trainer_protein_subset import ProteinSampleSubsetTrainer
from src.model.utils.data_collator import DataCollatorForT5Pssm

with open("../configs/model.yaml", "r") as f:
    train_config = yaml.safe_load(f)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


VERBOSE = train_config["verbose"]
SEED = train_config["seed"]

project_name = train_config["project_name"]
custom_run_name = train_config["custom_run_name"].replace(" ", "-")
model_name_identifier = (
    project_name
    + "-"
    + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    + (f"-{custom_run_name.replace(' ', '-')}" if custom_run_name else "")
)

USE_WANDB = train_config["weights_and_biases"]["enabled"]

if USE_WANDB:
    run = wandb.init(project=project_name, name=model_name_identifier)

print(model_name_identifier)




class PSSMHead(nn.Module):
    """Head for PSSM generation from T5 embeddings. based on https://github.com/hefeda/PGP/blob/master/prott5_batch_predictor.py#L144"""

    def __init__(self):
        """
        Args:
            config (MDPSSMConfig): Configuration object for the model
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv1d(1024, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(32, 20, kernel_size=7, padding=3),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.classifier(x)
        x = x.transpose(1, 2)
        pssm = torch.softmax(x, dim=2)
        return pssm


class T5EncoderModelForPssmGeneration(PreTrainedModel):
    def __init__(self, config: MDPSSMConfig):
        super().__init__(config=config)
        device_map = config.device if hasattr(config, "device") else "auto"
        plm_config = PLMConfig(
            name_or_path=config.model_name,
            device=device_map,
        )

        self.protein_encoder = ProstT5(config=plm_config)
        self.pssm_head = PSSMHead().to(device_map)
        self.loss_fct = KLDivLoss(reduction="batchmean")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # print(input_ids.shape)
        # print(attention_mask.shape)
        # print(attention_mask.sum())
        encoder_outputs = self.protein_encoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )

        # [batch_size, seq_len, hidden_dim]
        hidden_states = encoder_outputs["last_hidden_state"]
        attention_mask = encoder_outputs["mask"]

        # print(hidden_states.shape)
        # print(attention_mask.shape)
        # print(attention_mask.sum())
        # df = pd.DataFrame(hidden_states[0].cpu().numpy())
        # df.insert(0, "attention_mask", attention_mask[0].cpu().numpy())
        # df.insert(0, "input_ids", input_ids[0][1:].cpu().numpy())
        # display(df)

        # [batch_size, seq_len, 20]
        pssm = self.pssm_head(hidden_states)

        loss = None
        if labels is not None:
            # [batch_size * seq_len, 20]
            pred = pssm.flatten(end_dim=1)
            target = labels.flatten(end_dim=1)
            # print(target.shape)
            # print(pred.shape)

            pred_mask = attention_mask.flatten(end_dim=1)
            target_mask = ~torch.any(target == -100, dim=1)

            pred = pred[pred_mask.bool()]
            target = target[target_mask.bool()]

            # print(target.shape)
            # print(pred.shape)

            loss = self.loss_fct(torch.log(pred), target)
            # print(loss)

        if not return_dict:
            output = (pssm, encoder_outputs[2:-1])
            return ((loss,) + output) if loss is not None else output

        return PSSMOutput(
            loss=loss,
            pssms=pssm,
            hidden_states=encoder_outputs["last_hidden_state"] if output_hidden_states else None,
            masks=attention_mask,
        )


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

config = MDPSSMConfig(device=device, model_name="Rostlab/ProstT5")
model = T5EncoderModelForPssmGeneration(config)



import warnings
from Bio import BiopythonWarning
import json
import os
from tqdm import tqdm
import re

warnings.filterwarnings("ignore", category=BiopythonWarning)

MODEL_NAME = "prot-md-pssm-2025-04-02-15-20-18-dataset_320_0_prostt5"
SCOP40_SEQUENCES_FILE = "../tmp/data/scope/scope40_sequences.json"
MODEL_PATH = f"../tmp/models/adapters/{MODEL_NAME}"
PSSM_SAVE_DIR = f"../tmp/data/generated_pssms/scope40_{MODEL_NAME}"
PROTEIN_ENCODER_NAME = "Rostlab/prot_t5_xl_uniref50"

AA_ALPHABET = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
STRUCTURE_ALPHABET = [x.lower() for x in AA_ALPHABET]




with open(SCOP40_SEQUENCES_FILE, "r") as f:
    scop_sequences = json.load(f)
    # scop_sequences = dict(list(scop_sequences.items())[:11])
    scop_sequences = dict(list(scop_sequences.items()))

# for k, v in scop_sequences.items():
#     scop_sequences[k] = " ".join(list(re.sub(r"[UZOB]", "X", v)))

model.load_adapter(MODEL_PATH)
model.to(device)
model.eval()
print("Loaded model")



os.makedirs(PSSM_SAVE_DIR, exist_ok=True)

tokenizer = ProstT5Tokenizer()


def pssm_to_csv(name, pssm):
    df_pssm = pd.DataFrame(pssm)
    with open(f"{PSSM_SAVE_DIR}/{name}.tsv", "w") as f:
        f.write(f"Query profile of sequence {name}\n")
        f.write("     " + "      ".join(AA_ALPHABET) + "      \n")
        df_pssm = df_pssm.round(4)
        df_pssm.to_csv(f, index=False, sep=" ", float_format="%.4f", header=False, lineterminator=" \n")


batch_size = 20
sequence_items = list(scop_sequences.items())
sequence_batches = [dict(sequence_items[i : i + batch_size]) for i in range(0, len(sequence_items), batch_size)]


for batch in tqdm(sequence_batches, desc="Processing batches"):
    # print(batch.values())
    protein_tokens = tokenizer.encode(list(batch.values()), return_tensors="pt", padding=True, truncation=False).to(device)

    with torch.no_grad():
        model_output = model(
            input_ids=protein_tokens["input_ids"],
            attention_mask=protein_tokens["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
    torch.cuda.empty_cache()

    for name, pssm, mask, ids in zip(batch.keys(), model_output.pssms, model_output.masks, protein_tokens["input_ids"]):
        pssm = pssm[mask.cpu().numpy().astype(bool)].cpu().numpy()
        original_sequence = tokenizer.decode(ids, skip_special_tokens=True)  # .replace(" ", "")
        # print(name)
        # print(pssm.shape, mask.sum(), len(original_sequence))
        # print(*[f"{x:<4}" for x in original_sequence[1:]], sep="")
        # print(*[f"{x:<4}" for x in ids[1:]], sep="")
        # print(*[f"{x:<4}" for x in mask], sep="")
        # print(*[f"{x:<4}" for x in pssm.argmax(axis=1)], sep="")
        # print()
        # print(name, pssm.shape, len(original_sequence))
        pssm_to_csv(name, pssm)