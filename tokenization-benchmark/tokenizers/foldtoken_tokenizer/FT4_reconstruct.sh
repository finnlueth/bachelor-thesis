#! /bin/bash
export PYTHONPATH=$(pwd)/FoldToken_open_master/foldtoken
echo $PYTHONPATH

uv pip list

LEVEL=8
CUDA_VISIBLE_DEVICES=0

python FoldToken_open_master/foldtoken/reconstruct.py \
--path_in './data/casp15_targets_TSdomains_4invitees/' \
--path_out './data/reconstructed/' \
--level 9 \
--config './FoldToken_open_master/foldtoken/model_zoom/FT4/config.yaml' \
--checkpoint './FoldToken_open_master/foldtoken/model_zoom/FT4/ckpt.pth' 