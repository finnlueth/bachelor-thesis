#! /bin/bash
export PYTHONPATH=$(pwd)/FoldToken_open_master/foldtoken
echo $PYTHONPATH

uv pip list

LEVEL=8
CUDA_VISIBLE_DEVICES=0

python FoldToken_open_master/foldtoken/reconstruct_vqid.py \
--input_jsonl './data/tokenized/output_vqid_8_casp.jsonl' \
--path_out './data/detokenized/' \
--level 8 \
--config './FoldToken_open_master/foldtoken/model_zoom/FT4/config.yaml' \
--checkpoint './FoldToken_open_master/foldtoken/model_zoom/FT4/ckpt.pth'