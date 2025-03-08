#! /bin/bash

# make sure that data/input/ data/mat3di.out out/ pssms/ exist

FOLDSEEK=foldseek
MMSEQS=mmseqs
MATRIX=data/mat3di.out

VICTORS_PARAMETERS="--pca 1.4 --pcb 1.5 --sub-mat ${MATRIX} --mask-profile 0 --comp-bias-corr 0 --e-profile 0.1 -e 0.1 --profile-output-mode 1 --gap-open 11 --gap-extend 1"

"${FOLDSEEK}" createdb data/input/ out/inputdb
awk '{ len = $3 - 2; print "0\t"$1"\t0\t1.00\t0\t0\t"(len-1)"\t"len"\t0\t"(len-1)"\t"len"\t"len"M"; }' out/inputdb.index > out/fake_aln.tsv

"${MMSEQS}" tsv2db out/fake_aln.tsv out/fake_aln_db --output-dbtype 5
"${MMSEQS}" result2profile out/inputdb_ss out/inputdb_ss out/fake_aln_db pssm/profile.tsv ${VICTORS_PARAMETERS}
