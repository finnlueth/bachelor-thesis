#!/bin/bash

DATASET_TYPE="" # "_short" or ""

DATASET_IDENTIFIER=$1
# DATASET_IDENTIFIER=test${DATASET_TYPE}

FOLDSEEK="foldseek"
MMSEQS="mmseqs"

BENCHMARK="bench.noselfhit"
FOLDSEEK_ANALYSIS_SCOP_DIR=./scopbenchmark

SEQUENCE_FASTA_AA=../tmp/data/scope40/scope40_sequences_aa${DATASET_TYPE}.fasta
SEQUENCE_FASTA_3Di=../tmp/data/scope40/scope40_sequences_3Di${DATASET_TYPE}.fasta
LOOKUP_FILE=../tmp/data/scope40/scop_lookup.fix.tsv

PSSM_CSV=../tmp/data/pssm_generated/${DATASET_IDENTIFIER}.tsv
ROCX_DIR=../tmp/data/pssm_generated_rocx/
ROCX_FILE=${ROCX_DIR}/${DATASET_IDENTIFIER}.rocx

OUT_DIR=../tmp/out/foldseek_mdpssm_vs_3di_benchmark_${DATASET_IDENTIFIER}

ALIGN_DIR_RAW=${OUT_DIR}/alignResults/rawoutput
ALIGN_DIR_TMP=${OUT_DIR}/alignResults/tmp
ALIGN_DIR_ROCX=${OUT_DIR}/alignResults/rocx
DB_SEQUENCE=${OUT_DIR}/alignResults/db_sequence
DB_PROFILE=${OUT_DIR}/alignResults/db_profile
ALIGN_DIR_ALN=${OUT_DIR}/alignResults/tmp/aln

DB_NAME_MMSEQS=mmseqsDB
DB_NAME_FOLDSEEK=foldseekDB

if [ -d "${OUT_DIR}" ]; then
    rm -rf ${OUT_DIR}
fi

mkdir -p ${ROCX_DIR}
mkdir -p ${ALIGN_DIR_RAW}
mkdir -p ${ALIGN_DIR_TMP} 
mkdir -p ${ALIGN_DIR_ROCX}
mkdir -p ${DB_SEQUENCE}
mkdir -p ${DB_PROFILE}

date

echo "Running SCOPe40 benchmark"

echo "Generating FoldSeek DB"
python generate_foldseek_db_new.py ${SEQUENCE_FASTA_AA} ${SEQUENCE_FASTA_3Di} ${DB_SEQUENCE}/${DB_NAME_FOLDSEEK} ${OUT_DIR}

echo "Building PSSM DB"
python build_profiledb_new.py ${PSSM_CSV} ${DB_SEQUENCE}/${DB_NAME_FOLDSEEK} ${DB_PROFILE}

echo "Running FoldSeek search"
${FOLDSEEK} search ${DB_PROFILE}/${DB_NAME_FOLDSEEK}_profile ${DB_SEQUENCE}/${DB_NAME_FOLDSEEK} ${ALIGN_DIR_ALN} ${ALIGN_DIR_TMP} --threads 8 -s 9.5 --max-seqs 2000 -e 10

echo "Converting FoldSeek alignments"
${FOLDSEEK} convertalis ${DB_PROFILE}/${DB_NAME_FOLDSEEK}_profile ${DB_SEQUENCE}/${DB_NAME_FOLDSEEK} ${ALIGN_DIR_ALN} ${ALIGN_DIR_RAW}/foldseekaln

date

${FOLDSEEK_ANALYSIS_SCOP_DIR}/scripts/${BENCHMARK}.awk ${LOOKUP_FILE} <(cat ${ALIGN_DIR_RAW}/foldseekaln) > ${ROCX_FILE} #${ALIGN_DIR_ROCX}/foldseek.rocx

awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' ${ROCX_FILE} #${ALIGN_DIR_ROCX}/foldseek.rocx
