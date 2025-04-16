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

# identifiers_temperature = ["320", "348", "379", "413", "450"]
# identifiers_replica = ["0", "1", "2", "3", "4"]


temperature_identifier = 450
replica_identifier = "all"

dataset_identifier = temperature_identifier

config_yaml = f"""
metadata:
  name: "prot-md-pssm"
  identifier: 
  run_name: temp-{temperature_identifier}_repl-{replica_identifier}
  save_dir: ../tmp/models/adapters
  CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}
model:
#   encoder_name_or_path: Rostlab/prot_t5_xl_uniref50
  encoder_name_or_path: Rostlab/ProstT5
  hidden_size: 1024
  num_labels: 20
  dropout: 0.25
training_args:
  output_dir: ../tmp/models/checkpoints
  learning_rate: 0.0001
  per_device_train_batch_size: 10
  per_device_eval_batch_size: 10
  num_train_epochs: 1
  logging_steps: 1
  logging_strategy: steps
  evaluation_strategy: steps
  eval_steps: 32
  eval_strategy: steps
  eval_on_start: true
  batch_eval_metrics: false
  save_strategy: steps
  save_steps: 300
  save_total_limit: 5
  remove_unused_columns: true
  label_names: ['labels']
  seed: 42
  lr_scheduler_type: cosine
  warmup_steps: 0
  eval_sample_size: 30
lora:
  inference_mode: false
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  use_rslora: false
  use_dora: false
  target_modules: ['q', 'v']
  bias: none
data_collator:
  padding: true
  pad_to_multiple_of: 8
weights_and_biases:
  enabled: true
  project: prot-md-pssm
dataset:
  name: mdcath_pssm
  identifier: _{dataset_identifier}
  directory: ../tmp/data/pssm
"""

config = yaml.safe_load(config_yaml)

identifier = (
    config["metadata"]["name"]
    + "_"
    + config["model"]["encoder_name_or_path"].split("/")[-1]
    + "_"
    + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    + (f"_{config['metadata']['run_name'].replace(' ', '-')}" if config["metadata"]["run_name"] else "")
)
print(identifier)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(device)

if config["weights_and_biases"]["enabled"]:
    import wandb

    wandb.init(project=config["weights_and_biases"]["project"], name=identifier)
    run = wandb.init(project=config["weights_and_biases"]["project"], name=identifier)

ds = datasets.load_from_disk(f"{config['dataset']['directory']}/{config['dataset']['name']}{config['dataset']['identifier']}")
ds = ds.rename_column("pssm_features", "labels")

if config["model"]["encoder_name_or_path"] == "Rostlab/ProstT5":
    ds = ds.remove_columns(["input_ids_protT5", "attention_mask_protT5"])
    ds = ds.rename_column("input_ids_prostT5", "input_ids")
    ds = ds.rename_column("attention_mask_prostT5", "attention_mask")
if config["model"]["encoder_name_or_path"] == "Rostlab/prot_t5_xl_uniref50":
    ds = ds.remove_columns(["input_ids_prostT5", "attention_mask_prostT5"])
    ds = ds.rename_column("input_ids_protT5", "input_ids")
    ds = ds.rename_column("attention_mask_protT5", "attention_mask")

ds = ds.remove_columns(["name", "sequence", "replica", "temperature"])
# ds = ds.select(range(25))  # !!! TODO REMOVE THIS !!!
print(ds)

model_config = PLMConfigForPSSM(**config["model"])
model = PLMForPssmGeneration(model_config)
model.to(device)
print(model)

lora_config = LoraConfig(**config["lora"], modules_to_save=model.get_modules_to_save())
model = get_peft_model(model, lora_config)
print("target_modules:", lora_config.target_modules)
print("modules_to_save:", lora_config.modules_to_save)
model.print_trainable_parameters()

tokenizer = auto_tokenizer(config["model"]["encoder_name_or_path"])

data_collator = DataCollatorForPSSM(
    tokenizer=tokenizer.get_tokenizer(),
    padding=True,
    pad_to_multiple_of=8,
)


class CustomTrainingArguments(TrainingArguments):
    def __init__(self, eval_sample_size=32, **kwargs):
        self.eval_sample_size = eval_sample_size
        super().__init__(**kwargs)


config["training_args"]["eval_sample_size"] = config["training_args"].get("eval_sample_size", 32)

training_args = CustomTrainingArguments(**config["training_args"])

trainer = ProteinSampleSubsetTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    eval_dataset=ds,
    data_collator=data_collator,
)


def clean_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


clean_memory()
trainer.train()
trainer.evaluate()
clean_memory()


model_save_path = f"{config['metadata']['save_dir']}/{identifier}"

model.save_pretrained(save_directory=model_save_path)

pd.DataFrame(trainer.state.log_history).to_csv(f"{model_save_path}/training_log.csv", index=False)

with open(f"{model_save_path}/train_config.yaml", "w") as f:
    config["metadata"]["identifier"] = identifier
    yaml.dump(config, f, sort_keys=False)


def plot_training_history(log_history, metrics_names=["loss", "eval_loss"]):
    plt.style.use("default")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    train_logs = log_history[log_history["loss"].notna()]
    eval_logs = log_history[log_history["eval_loss"].notna()]

    ax1.plot(train_logs["epoch"], train_logs["loss"], label="Training Loss", color="orange", linewidth=1)
    ax1.plot(eval_logs["epoch"], eval_logs["eval_loss"], label="Eval Loss", color="lightblue", linewidth=1)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("KL Divergence Loss", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.legend(loc="upper right")
    ax1.grid(True)
    plt.tight_layout()
    return fig


fig = plot_training_history(log_history=pd.DataFrame(trainer.state.log_history), metrics_names=["loss"])
fig.savefig(f"{model_save_path}/training_history.png")
plt.close(fig)

print("Model, config, and log saved to:", model_save_path)

# ----------------------------------------------------------------------------------------------------------------------

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
