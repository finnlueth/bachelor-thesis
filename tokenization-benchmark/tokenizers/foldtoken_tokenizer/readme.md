# readme


## install
```sh
uv venv
source .venv/bin/activate
uv sync
```

```sh
export PYTHONPATH='/home/finnlueth/repos/bachelor-thesis/tokenization-benchmark/tokenizers/foldtoken_tokenizer/FoldToken_open_master/foldtoken'
CUDA_VISIBLE_DEVICES=0 python extract_vq_ids.py --path_in '/home/finnlueth/repos/bachelor-thesis/tokenization-benchmark/tokenizers/foldtoken_tokenizer/data/casp15_targets_TSdomains_4invitees' --save_vqid_path ./output_vqid.jsonl --level 8
```
