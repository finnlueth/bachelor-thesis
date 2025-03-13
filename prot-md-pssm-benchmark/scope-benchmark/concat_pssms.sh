# # Replace 0 in header with filename for each PSSM file
# for file in ./data/scope40_prot-md-pssm-2025-03-05-17-43-47-full-dataset/*.tsv; do
#     # Get filename without extension
#     echo "Processing $file"
#     filename=$(basename "$file" .tsv)
#     # Replace first line (header) 0 with filename only if header is "Query profile of sequence 0"
#     sed -i '1s/^Query profile of sequence 0$/Query profile of sequence '"$filename"'/' "$file"
# done



cat ./data/scope40_prot-md-pssm-2025-03-05-17-43-47-full-dataset/*.tsv > ./data/scope40_prot-md-pssm-2025-03-05-17-43-47-full-dataset_concatenated.tsv

# ls ./data/scope40_prot-md-pssm-2025-03-05-17-43-47-full-dataset/*.tsv | head -n 40 | xargs cat > ./data/scope40_prot-md-pssm-2025-03-05-17-43-47-full-dataset_concatenated_short.tsv