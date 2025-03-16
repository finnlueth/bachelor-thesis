#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3
FOLDSEEK="foldseek"
RUN_FOLDSEEK=true

BENCHMARK="bench.noselfhit"

FOLDSEEK_ANALYSIS_SCOP_DIR=../foldseek-analysis/scopbenchmark

LOOKUP_FILE=${FOLDSEEK_ANALYSIS_SCOP_DIR}/data/scop_lookup.fix.tsv
SEQUENCE_FASTA=./data/scope40_sequences.fasta

OUT_DIR=./out/foldseek_3Di_benchmark
ALIGN_DIR_RAW=${OUT_DIR}/alignResults/rawoutput
ALIGN_DIR_TMP=${OUT_DIR}/alignResults/tmp
ALIGN_DIR_ROCX=${OUT_DIR}/alignResults/rocx
ALIGN_DIR_ALN=${OUT_DIR}/alignResults/tmp/aln

mkdir -p ${ALIGN_DIR_RAW}
mkdir -p ${ALIGN_DIR_TMP} 
mkdir -p ${ALIGN_DIR_ROCX}
mkdir -p ${ALIGN_DIR_ALN}

## start timing
date 
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
if [ "$RUN_FOLDSEEK" = true ]; then
    ${FOLDSEEK} createdb ${SEQUENCE_FASTA} ${ALIGN_DIR_TMP}/queryDB --prostt5-model ./data/prostt5/prostt5_out/prostt5-f16.gguf --gpu 1
    ${FOLDSEEK} search ${ALIGN_DIR_TMP}/queryDB ${ALIGN_DIR_TMP}/queryDB ${ALIGN_DIR_ALN} ${ALIGN_DIR_TMP} --threads 8 -s 9.5 --max-seqs 2000 -e 10
    ${FOLDSEEK} convertalis ${ALIGN_DIR_TMP}/queryDB ${ALIGN_DIR_TMP}/queryDB ${ALIGN_DIR_ALN} ${ALIGN_DIR_RAW}/foldseekaln
else
    echo "Foldseek not run, attempting to use existing alignments"
fi

## end timing
date

## generate ROCX file
${FOLDSEEK_ANALYSIS_SCOP_DIR}/scripts/${BENCHMARK}.awk ${LOOKUP_FILE} <(cat ${ALIGN_DIR_RAW}/foldseekaln) > ${ALIGN_DIR_ROCX}/foldseek.rocx

## calculate auc
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' ${ALIGN_DIR_ROCX}/foldseek.rocx
