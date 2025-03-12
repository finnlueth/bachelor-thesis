    # Create MMseqs2 databases for query and target
    ${MMSEQS} createdb ${SEQUENCE_FASTA} ${DB_DIR}/queryDB --dbtype 12
    ${MMSEQS} createdb ${SEQUENCE_FASTA} ${DB_DIR}/targetDB --dbtype 12
    
    # Create index for the target database
    ${MMSEQS} createindex ${DB_DIR}/targetDB ${ALIGN_DIR_TMP} --remove-tmp-files 1
    
    # Perform the search
    ${MMSEQS} search ${DB_DIR}/queryDB ${DB_DIR}/targetDB ${DB_DIR}/resultDB ${ALIGN_DIR_TMP} \
        -a --threads 8 -s 7.5 -e 10000 --max-seqs 2000
    
    # Convert results to readable format
    ${MMSEQS} convertalis ${DB_DIR}/queryDB ${DB_DIR}/targetDB ${DB_DIR}/resultDB ${ALIGN_DIR_RAW}/mmseqsaln \
        --format-output "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"