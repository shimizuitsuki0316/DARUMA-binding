#!/bin/bash

fasta_file="sample_data/DP03914.fasta"
embeddings_file="sample_data/DP03914.npy"
output_file="sample_data/DP03914.caid"

# 実行
python3 daruma_binding/predict.py \
    --fasta $fasta_file \
    --embeddings $embeddings_file \
    --output $output_file

