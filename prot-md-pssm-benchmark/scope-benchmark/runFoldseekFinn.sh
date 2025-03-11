#!/bin/bash

FOLDSEEK="foldseek"
RUN_FOLDSEEK=false

BENCHMARK="bench.noselfhit"

FOLDSEEK_ANALYSIS_SCOP_DIR=../foldseek-analysis/scopbenchmark

LOOKUP_FILE=${FOLDSEEK_ANALYSIS_SCOP_DIR}/data/scop_lookup.fix.tsv
PDB_DIR=${FOLDSEEK_ANALYSIS_SCOP_DIR}/data/scop-pdb

OUT_DIR=./out/foldseek_benchmark/
ALIGN_DIR_RAW=${OUT_DIR}/alignResults/rawoutput/
ALIGN_DIR_TMP=${OUT_DIR}/alignResults/tmp
ALIGN_DIR_ROCX=${OUT_DIR}/alignResults/rocx

mkdir -p ${ALIGN_DIR_RAW}
mkdir -p ${ALIGN_DIR_TMP} 
mkdir -p ${ALIGN_DIR_ROCX}

## start timing
date 

## foldseek easy-search (combination of all convert2db + prefilter + align)
if [ "$RUN_FOLDSEEK" = true ]; then
    ${FOLDSEEK} easy-search ${PDB_DIR}/ ${PDB_DIR}/ ${ALIGN_DIR_RAW}/foldseekaln ${ALIGN_DIR_TMP}/ --threads 8 -s 9.5 --max-seqs 2000 -e 10
else
    echo "Foldseek not run, attempting to use existing alignments"
fi

## end timing
date

## generate ROCX file
${FOLDSEEK_ANALYSIS_SCOP_DIR}/scripts/${BENCHMARK}.awk ${LOOKUP_FILE} <(cat ${ALIGN_DIR_RAW}/foldseekaln) > ${ALIGN_DIR_ROCX}/foldseek.rocx

## calculate auc
 awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' ${ALIGN_DIR_ROCX}/foldseek.rocx