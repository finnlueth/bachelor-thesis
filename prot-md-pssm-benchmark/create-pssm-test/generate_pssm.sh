FOLDSEEK=foldseek
MMSEQS=mmseqs
MAT3DI_OUT=./data/mat3di.out

INPUT_DIR=./data/md-pdbs
FOLDSEEK_OUT=./data/out
PSSM_OUT=./data/pssm

mkdir -p ${INPUT_DIR}
mkdir -p ${FOLDSEEK_OUT}
mkdir -p ${PSSM_OUT}

VICTORS_PARAMETERS="--pca 1.4 --pcb 1.5 --sub-mat ${MAT3DI_OUT} --mask-profile 0 --comp-bias-corr 0 --e-profile 0.1 -e 0.1 --profile-output-mode 1 --gap-open 11 --gap-extend 1"

"${FOLDSEEK}" createdb ${INPUT_DIR}/ ${FOLDSEEK_OUT}/inputdb

awk '{ len = $3 - 2; print "0\t"$1"\t0\t1.00\t0\t0\t"(len-1)"\t"len"\t0\t"(len-1)"\t"len"\t"len"M"; }' ${FOLDSEEK_OUT}/inputdb.index > ${FOLDSEEK_OUT}/fake_aln.tsv

"${MMSEQS}" tsv2db ${FOLDSEEK_OUT}/fake_aln.tsv ${FOLDSEEK_OUT}/fake_aln_db --output-dbtype 5
"${MMSEQS}" result2profile ${FOLDSEEK_OUT}/inputdb_ss ${FOLDSEEK_OUT}/inputdb_ss ${FOLDSEEK_OUT}/fake_aln_db ${PSSM_OUT}/profile.tsv ${VICTORS_PARAMETERS}
