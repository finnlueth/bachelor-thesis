#!/usr/bin/env python3

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import gc
import json
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime

import datasets
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from peft import LoraConfig, get_peft_model
from plms import auto_tokenizer
from tqdm import tqdm
from transformers import TrainingArguments

from plm_pssms import DataCollatorForPSSM, PLMConfigForPSSM, PLMForPssmGeneration, ProteinSampleSubsetTrainer

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(device)

identifier = "prot-md-pssm_ProstT5_2025-04-16-19-42-44_temp-450_repl-all"

model_save_path = f"../tmp/models/adapters/{identifier}"

with open(f"{model_save_path}/train_config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_config = PLMConfigForPSSM(**config["model"])
model = PLMForPssmGeneration(model_config)
model.load_adapter(model_save_path)
model.to(device)
tokenizer = auto_tokenizer(config["model"]["encoder_name_or_path"])



scope40_seq_file_path = "../tmp/data/scope40/scope40_sequences.json"
pssm_save_path = f"../tmp/data/pssm_generated/{identifier}.tsv"

AA_ALPHABET = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
STRUCTURE_ALPHABET = [x.lower() for x in AA_ALPHABET]

with open(scope40_seq_file_path, "r") as f:
    scop_sequences = json.load(f)
    # scop_sequences = dict(list(scop_sequences.items())[:111]) # !!! TODO REMOVE THIS !!!


def pssm_to_csv(name, pssm):
    df_pssm = pd.DataFrame(pssm)
    df_pssm = df_pssm.round(4)

    tsv_string = f"Query profile of sequence {name}\n"
    tsv_string += "     " + "      ".join(AA_ALPHABET) + "      \n"

    df_string = df_pssm.to_csv(index=False, sep=" ", float_format="%.4f", header=False, lineterminator=" \n")
    tsv_string += df_string

    return tsv_string


batch_size = 20
pssm_tsv = ""
sequence_items = list(scop_sequences.items())
sequence_batches = [dict(sequence_items[i : i + batch_size]) for i in range(0, len(sequence_items), batch_size)]

if os.path.exists(pssm_save_path):
    os.remove(pssm_save_path)

model.eval()
for batch in tqdm(sequence_batches, desc="Processing batches"):
    pssm_tsv = ""
    protein_tokens = tokenizer.encode(list(batch.values()), return_tensors="pt", padding=True, truncation=False).to(device)

    with torch.no_grad():
        model_output = model(
            input_ids=protein_tokens["input_ids"],
            attention_mask=protein_tokens["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
    torch.cuda.empty_cache()

    for name, pssm, mask, ids in list(zip(batch.keys(), model_output.pssms, model_output.masks, protein_tokens["input_ids"])):
        pssm = pssm[mask.cpu().numpy().astype(bool)].cpu().numpy()
        original_sequence = tokenizer.decode(ids, skip_special_tokens=True)
        pssm_tsv += pssm_to_csv(name, pssm)

    with open(pssm_save_path, "a") as f:
        f.write(pssm_tsv)

print("Created and appended PSSM to:", pssm_save_path)

# ----------------------------------------------------------------------------------------------------------------------

print("Current working directory:", os.getcwd())


@contextmanager
def working_directory(path):
    """Temporarily change working directory."""
    previous_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous_dir)


with working_directory("../benchmark"):
    print("Current working directory:", os.getcwd())
    benchmark_script = "./runFoldseekMDPSSM.sh"

    print(f"Running benchmark with dataset ID: {identifier}")

    try:
        result = subprocess.run([benchmark_script, identifier], check=True, text=True, capture_output=True)
        print("Benchmark output:")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print("Benchmark failed with error:")
        print(e.stderr)
print("Current working directory:", os.getcwd())
