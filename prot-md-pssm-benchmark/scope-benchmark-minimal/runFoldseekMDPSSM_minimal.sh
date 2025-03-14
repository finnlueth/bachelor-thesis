#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3
FOLDSEEK="foldseek"
MMSEQS="mmseqs"

BENCHMARK="bench.noselfhit"

FOLDSEEK_ANALYSIS_SCOP_DIR=./scopbenchmark

SEQUENCE_FASTA_AA=./data/scope40_sequences_short.fasta
SEQUENCE_FASTA_3Di=./data/scope40_sequences_3Di_short.fasta
PSSM_CSV=./data/scope40_prot-md-pssm-2025-03-05-17-43-47-full-dataset_concatenated_short.tsv
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

## start timing
date

echo "Running Foldseek or MMseqs"

# Create the 3Di and AA databases
python generate_foldseek_db.py ${SEQUENCE_FASTA_AA} ${SEQUENCE_FASTA_3Di} ${DB_SEQUENCE}/${DB_NAME_FOLDSEEK} ${OUT_DIR}

# Build the profile database from PSSMs, this creates targetDB_profile and targetDB_profile
# From Victor: Here's a script to write a foldseek profiledb from tsv 3DI profiles and an amino acid sequence database. You can then use this resulting database as a query for any foldseek search.
# From Victor: The first argument is the tsv file, the second the amino acid sequence db and the third the output directory. For your scop benchmark I think you can replace the scop query with the resulting profiledb in the search against itself
# Question: Line 276 in build_profiledb.py: `shutil.copy2(src_lookup, dest_lookup)` requires lookup file to exist, which is not created in the generate_foldseek_db.py script.
# Do I need the lookup file? How do I get it? 
# When commenting out the line, the script runs fine until convertalis.
python build_profiledb.py ${PSSM_CSV} ${DB_SEQUENCE}/${DB_NAME_FOLDSEEK} ${DB_PROFILE}

# Run the search using the profile database as query against the sequence database as target
# Profile vs Sequence
${FOLDSEEK} search ${DB_PROFILE}/${DB_NAME_FOLDSEEK}_profile ${DB_SEQUENCE}/${DB_NAME_FOLDSEEK} ${ALIGN_DIR_ALN} ${ALIGN_DIR_TMP} --threads 8 -s 9.5 --max-seqs 2000 -e 10

# Convert the alignment
${FOLDSEEK} convertalis ${DB_PROFILE}/${DB_NAME_FOLDSEEK}_profile ${DB_SEQUENCE}/${DB_NAME_FOLDSEEK} ${ALIGN_DIR_ALN} ${ALIGN_DIR_RAW}/foldseekaln


# end timing
date

# generate ROCX file
${FOLDSEEK_ANALYSIS_SCOP_DIR}/scripts/${BENCHMARK}.awk ${LOOKUP_FILE} <(cat ${ALIGN_DIR_RAW}/foldseekaln) > ${ALIGN_DIR_ROCX}/foldseek.rocx

# calculate auc
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' ${ALIGN_DIR_ROCX}/foldseek.rocx
