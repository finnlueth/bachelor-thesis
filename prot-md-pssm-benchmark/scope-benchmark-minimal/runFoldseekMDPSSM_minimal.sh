#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3
FOLDSEEK="foldseek"
MMSEQS="mmseqs"

BENCHMARK="bench.noselfhit"
FOLDSEEK_ANALYSIS_SCOP_DIR=./scopbenchmark

DATASET_TYPE=""
SEQUENCE_FASTA_AA=./data/scope40_sequences_aa${DATASET_TYPE}.fasta
SEQUENCE_FASTA_3Di=./data/scope40_sequences_3Di_argmax${DATASET_TYPE}.fasta
PSSM_CSV=./data/scope40_prot-md-pssm-2025-03-05-17-43-47-full-dataset_concatenated${DATASET_TYPE}.tsv
LOOKUP_FILE=./data/scop_lookup.fix.tsv

OUT_DIR=./out/foldseek_mdpssm_benchmark
ALIGN_DIR_RAW=${OUT_DIR}/alignResults/rawoutput
DB_SEQUENCE=${OUT_DIR}/alignResults/db_sequence
DB_PROFILE=${OUT_DIR}/alignResults/db_profile
ALIGN_DIR_TMP=${OUT_DIR}/alignResults/tmp
ALIGN_DIR_ROCX=${OUT_DIR}/alignResults/rocx
ALIGN_DIR_ALN=${OUT_DIR}/alignResults/tmp/aln
DB_NAME_MMSEQS=mmseqsDB
DB_NAME_FOLDSEEK=foldseekDB

if [ -d "${OUT_DIR}" ]; then
    rm -rf ${OUT_DIR}
fi

mkdir -p ${ALIGN_DIR_RAW}
mkdir -p ${ALIGN_DIR_TMP} 
mkdir -p ${ALIGN_DIR_ROCX}
mkdir -p ${DB_SEQUENCE}
mkdir -p ${DB_PROFILE}

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

date

echo "Running Foldseek or MMseqs"

python generate_foldseek_db_new.py ${SEQUENCE_FASTA_AA} ${SEQUENCE_FASTA_3Di} ${DB_SEQUENCE}/${DB_NAME_FOLDSEEK} ${OUT_DIR}

python build_profiledb_new.py ${PSSM_CSV} ${DB_SEQUENCE}/${DB_NAME_FOLDSEEK} ${DB_PROFILE}

${FOLDSEEK} search ${DB_PROFILE}/${DB_NAME_FOLDSEEK}_profile ${DB_SEQUENCE}/${DB_NAME_FOLDSEEK} ${ALIGN_DIR_ALN} ${ALIGN_DIR_TMP} --threads 8 -s 9.5 --max-seqs 2000 -e 10

${FOLDSEEK} convertalis ${DB_PROFILE}/${DB_NAME_FOLDSEEK}_profile ${DB_SEQUENCE}/${DB_NAME_FOLDSEEK} ${ALIGN_DIR_ALN} ${ALIGN_DIR_RAW}/foldseekaln

date

${FOLDSEEK_ANALYSIS_SCOP_DIR}/scripts/${BENCHMARK}.awk ${LOOKUP_FILE} <(cat ${ALIGN_DIR_RAW}/foldseekaln) > ${ALIGN_DIR_ROCX}/foldseek.rocx

awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' ${ALIGN_DIR_ROCX}/foldseek.rocx
