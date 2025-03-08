#!/bin/bash

## start timing
date 

## foldseek easy-search (combination of all convert2db + prefilter + align)
# foldseek easy-search ../data/scop-pdb/ ../data/scop-pdb/ ../alignResults/rawoutput/foldseekaln ../alignResults/tmp/ --threads 8 -s 9.5 --max-seqs 2000 -e 10

## end timing
date

## generate ROCX file
mawk -f bench.noselfhit.awk ../data/scop_lookup_short.fix.tsv <(cat ../alignResults/rawoutput/foldseekaln) > ../alignResults/rocx/foldseek.rocx

## calculate auc
awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' ../alignResults/rocx/foldseek.rocx


