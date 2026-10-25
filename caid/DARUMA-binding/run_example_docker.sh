#!/bin/bash

docker run --rm \
  -v "$PWD/sample_data":/work \
  daruma-binding:v0.1.0 \
  --fasta /work/DP03914.fasta \
  --embeddings /work/DP03914.npy \
  --output /work/DP03914.caid \
  --device cpu