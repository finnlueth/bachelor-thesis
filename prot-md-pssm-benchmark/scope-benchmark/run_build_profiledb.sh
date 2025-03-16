PSSM_CSV=./data/scope40_prot-md-pssm-2025-03-05-17-43-47-full-dataset_concatenated_short.tsv
MMSEQS_DB=./out/foldseek_mdpssm_benchmark/mmseqs_db
OUT_DIR=./out/foldseek_mdpssm_benchmark/out_dir

mkdir -p $OUT_DIR

python build_profiledb.py $PSSM_CSV $MMSEQS_DB $OUT_DIR