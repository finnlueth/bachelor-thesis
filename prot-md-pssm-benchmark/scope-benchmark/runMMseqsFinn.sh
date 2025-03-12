#!/bin/bash

MMSEQS="mmseqs"
RUN_MMSEQS=true

BENCHMARK="bench.noselfhit"

FOLDSEEK_ANALYSIS_SCOP_DIR=../foldseek-analysis/scopbenchmark

LOOKUP_FILE=${FOLDSEEK_ANALYSIS_SCOP_DIR}/data/scop_lookup.fix.tsv
SEQUENCE_FASTA=./data/scope40_sequences.fasta

OUT_DIR=./out/mmseqs_benchmark/
ALIGN_DIR_RAW=${OUT_DIR}/alignResults/rawoutput/
ALIGN_DIR_TMP=${OUT_DIR}/alignResults/tmp
ALIGN_DIR_ROCX=${OUT_DIR}/alignResults/rocx

mkdir -p ${ALIGN_DIR_RAW}
mkdir -p ${ALIGN_DIR_TMP} 
mkdir -p ${ALIGN_DIR_ROCX}

## start timing
date 

## mmseqs search module
if [ "$RUN_MMSEQS" = true ]; then
    ${MMSEQS} easy-search ${SEQUENCE_FASTA} ${SEQUENCE_FASTA} ${ALIGN_DIR_RAW}/mmseqsaln ${ALIGN_DIR_TMP} -a --threads 8 -s 7.5 -e 10000 --max-seqs 2000
else
    echo "MMseqs not run, attempting to use existing alignments"
fi

## end timing
date


## generate ROCX file
${FOLDSEEK_ANALYSIS_SCOP_DIR}/scripts/${BENCHMARK}.awk ${LOOKUP_FILE} <(cat ${ALIGN_DIR_RAW}/mmseqsaln) > ${ALIGN_DIR_ROCX}/mmseqs.rocx

## calculate auc
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' ${ALIGN_DIR_ROCX}/mmseqs.rocx
