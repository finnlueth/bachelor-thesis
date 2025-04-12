#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3
FOLDSEEK="foldseek"
MMSEQS="mmseqs"
RUN_FOLDSEEK=true

BENCHMARK="bench.noselfhit"

FOLDSEEK_ANALYSIS_SCOP_DIR=../foldseek-analysis/scopbenchmark

SEQUENCE_FASTA_AA=./data/scope40_sequences_short.fasta
SEQUENCE_FASTA_3Di=./data/scope40_sequences_3Di_short.fasta
PSSM_CSV=./data/scope40_prot-md-pssm-2025-03-05-17-43-47-full-dataset_concatenated_short.tsv
LOOKUP_FILE=${FOLDSEEK_ANALYSIS_SCOP_DIR}/data/scop_lookup.fix.tsv

OUT_DIR=./out/foldseek_prott5_mdpssm_benchmark
ALIGN_DIR_RAW=${OUT_DIR}/alignResults/rawoutput
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

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

## start timing
date


if [ "$RUN_FOLDSEEK" = true ]; then
    echo "Running Foldseek or MMseqs"
    # ------------------------------------------------------------------------------------------------
    # First create the sequence database that will be used as target
    
    # echo "Creating sequence databases for 3Di"
    # ${MMSEQS} createdb ${SEQUENCE_FASTA_3Di} ${ALIGN_DIR_TMP}/${DB_NAME_FOLDSEEK}_ss
    # echo "Creating sequence databases for AA"
    # ${MMSEQS} createdb ${SEQUENCE_FASTA_AA} ${ALIGN_DIR_TMP}/${DB_NAME_FOLDSEEK}

    python generate_foldseek_db.py ${SEQUENCE_FASTA_AA} ${SEQUENCE_FASTA_3Di} ${ALIGN_DIR_TMP}/${DB_NAME_FOLDSEEK} ${OUT_DIR}
    cp ${ALIGN_DIR_TMP}/${DB_NAME_FOLDSEEK}_h ${ALIGN_DIR_TMP}/${DB_NAME_FOLDSEEK}_ss_h
    cp ${ALIGN_DIR_TMP}/${DB_NAME_FOLDSEEK}_h.dbtype ${ALIGN_DIR_TMP}/${DB_NAME_FOLDSEEK}_ss_h.dbtype
    cp ${ALIGN_DIR_TMP}/${DB_NAME_FOLDSEEK}_h.index ${ALIGN_DIR_TMP}/${DB_NAME_FOLDSEEK}_ss_h.index


    # ${FOLDSEEK} createdb ${ALIGN_DIR_TMP}/${MMSEQS_DB_NAME} ${ALIGN_DIR_TMP}/${FOLDSEEK_DB_NAME}

    # ${FOLDSEEK} createdb ${SEQUENCE_FASTA} ${ALIGN_DIR_TMP}/${FOLDSEEK_DB_NAME}
    # ------------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------------
    # Build the profile database from PSSMs - this will be our query database
    # This creates targetDB_profile and targetDB_profile which we'll use as query
    # Victor: Here's a script to write a foldseek profiledb from tsv 3DI profiles and an amino acid sequence database. You can then use this resulting database as a query for any foldseek search.
    # Victor: The first argument is the tsv file, the second the amino acid sequence db and the third the output directory. For your scop benchmark I think you can replace the scop query with the resulting profiledb in the search against itself
    
    echo "Building profile database"
    python build_profiledb.py ${PSSM_CSV} ${ALIGN_DIR_TMP}/${DB_NAME_FOLDSEEK} ${ALIGN_DIR_TMP}
    # ------------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------------
    # Run the search using the profile database as query against the sequence database as target
    
    # Profile vs Profile
    # ${FOLDSEEK} search ${ALIGN_DIR_TMP}/${DB_NAME}_profile ${ALIGN_DIR_TMP}/${DB_NAME} ${ALIGN_DIR_ALN} ${ALIGN_DIR_TMP} --threads 8 -s 9.5 --max-seqs 2000 -e 10
    
    # Profile vs Sequence
    ${FOLDSEEK} search ${ALIGN_DIR_TMP}/${DB_NAME_FOLDSEEK}_profile ${ALIGN_DIR_TMP}/${DB_NAME_FOLDSEEK} ${ALIGN_DIR_ALN} ${ALIGN_DIR_TMP} --threads 8 -s 9.5 --max-seqs 2000 -e 10
    
    # Sequence vs Profile
    # ${FOLDSEEK} search ${ALIGN_DIR_TMP}/${DB_NAME} ${ALIGN_DIR_TMP}/${DB_NAME}_profile ${ALIGN_DIR_ALN} ${ALIGN_DIR_TMP} --threads 8 -s 9.5 --max-seqs 2000 -e 10
    # ------------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------------
    ${FOLDSEEK} convertalis ${ALIGN_DIR_TMP}/${DB_NAME_FOLDSEEK}_profile ${ALIGN_DIR_TMP}/${DB_NAME_FOLDSEEK} ${ALIGN_DIR_ALN} ${ALIGN_DIR_RAW}/foldseekaln
    # ------------------------------------------------------------------------------------------------
else
    echo "Foldseek not run, attempting to use existing alignments"
fi

## end timing
date

# generate ROCX file
${FOLDSEEK_ANALYSIS_SCOP_DIR}/scripts/${BENCHMARK}.awk ${LOOKUP_FILE} <(cat ${ALIGN_DIR_RAW}/foldseekaln) > ${ALIGN_DIR_ROCX}/foldseek.rocx

## calculate auc
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' ${ALIGN_DIR_ROCX}/foldseek.rocx
