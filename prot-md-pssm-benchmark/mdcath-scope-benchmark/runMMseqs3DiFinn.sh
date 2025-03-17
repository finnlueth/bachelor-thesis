#!/bin/bash

FOLDSEEK="mmseqs"
RUN_FOLDSEEK=true

BENCHMARK="bench.noselfhit"

FOLDSEEK_ANALYSIS_SCOP_DIR=./scopbenchmark

LOOKUP_FILE=./data/mdCATH_lookup.fix.tsv
SEQUENCE_FASTA=./data/mdCATH_sequence_3Di.fasta

OUT_DIR=./out/foldseek_3Di_as_AA_vs_3Di_as_AA_benchmark/
ALIGN_DIR_RAW=${OUT_DIR}/alignResults/rawoutput/
ALIGN_DIR_TMP=${OUT_DIR}/alignResults/tmp
ALIGN_DIR_ROCX=${OUT_DIR}/alignResults/rocx

mkdir -p ${ALIGN_DIR_RAW}
mkdir -p ${ALIGN_DIR_TMP} 
mkdir -p ${ALIGN_DIR_ROCX}

## start timing
date 

## foldseek search module
if [ "$RUN_FOLDSEEK" = true ]; then
    ${FOLDSEEK} easy-search ${SEQUENCE_FASTA} ${SEQUENCE_FASTA} ${ALIGN_DIR_RAW}/foldseekaln ${ALIGN_DIR_TMP} --threads 8 -s 9.5 --max-seqs 2000 -e 10
else
    echo "Foldseek not run, attempting to use existing alignments"
fi



## end timing
date


## generate ROCX file
${FOLDSEEK_ANALYSIS_SCOP_DIR}/scripts/${BENCHMARK}.awk ${LOOKUP_FILE} <(cat ${ALIGN_DIR_RAW}/foldseekaln) > ${ALIGN_DIR_ROCX}/foldseek.rocx

## calculate auc
awk '{ classsum+=$3; archsum+=$4; topsum+=$5; homsum+=$6}END{print classsum/NR,archsum/NR,topsum/NR,homsum/NR}' ${ALIGN_DIR_ROCX}/mmseqs.rocx
