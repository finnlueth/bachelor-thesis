#!/bin/bash

MMSEQS="mmseqs"
RUN_MMSEQS=true

BENCHMARK="bench.noselfhit"

FOLDSEEK_ANALYSIS_SCOP_DIR=./scopbenchmark

LOOKUP_FILE=./data/mdCATH_lookup.fix.tsv
SEQUENCE_FASTA=./data/mdCATH_sequence_AA.fasta

OUT_DIR=./out/mmseqs_benchmark_aa_aa/
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
awk '{ classsum+=$3; archsum+=$4; topsum+=$5; homsum+=$6}END{print classsum/NR,archsum/NR,topsum/NR,homsum/NR}' ${ALIGN_DIR_ROCX}/mmseqs.rocx
