FOLDSEEK=foldseek
MMSEQS=mmseqs
MAT3DI_OUT=../data/mat3di.out
VICTORS_PARAMETERS="--pca 1.4 --pcb 1.5 --sub-mat ${MAT3DI_OUT} --mask-profile 0 --comp-bias-corr 0 --e-profile 0.1 -e 0.1 --profile-output-mode 1 --gap-open 11 --gap-extend 1"

"${FOLDSEEK}" createdb input/ inputdb

awk '{ len = $3 - 2; print "0\t"$1"\t0\t1.00\t0\t0\t"(len-1)"\t"len"\t0\t"(len-1)"\t"len"\t"len"M"; }' inputdb.index > fake_aln.tsv

"${MMSEQS}" tsv2db fake_aln.tsv fake_aln_db --output-dbtype 5
"${MMSEQS}" result2profile inputdb_ss inputdb_ss fake_aln_db profile.tsv ${VICTORS_PARAMETERS}





