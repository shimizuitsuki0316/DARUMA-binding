# DARUMA-binding

DARUMA-binding is a predictor for binding residues in intrinsically disordered regions.

## Overview

DARUMA-binding predicts binding residues in intrinsically disordered regions using precomputed ProtT5 embeddings. 
The model outputs residue-wise binding probabilities and binary predictions in CAID format.

## Input

DARUMA-binding requires:

1. A FASTA file
2. A precomputed ProtT5 embedding file in NumPy `.npy` format

The embedding array must have shape `L × 1024`, where `L` is the protein sequence length.

## Protein language model

DARUMA-binding uses precomputed per-residue ProtT5 embeddings.

- Protein language model: `Rostlab/prot_t5_xl_half_uniref50-enc`
- Embedding type: per-residue embedding
- Embedding dimension: 1024
- Input format: NumPy `.npy`
- Embedding shape: `L × 1024`, where `L` is the sequence length
- Parameters for embedding generation: default parameters
- Fine-tuning of ProtT5: none
- ProtT5 model and weights: not included in this Docker image

## Run with Docker
```bash
docker pull daruma-binding:v0.1.0
docker run --rm \
  -v "$PWD/sample_data":/work \
  daruma-binding:v0.1.0 \
  --fasta /work/DP03914.fasta \
  --embeddings /work/DP03914.npy \
  --output /work/DP03914.caid \
  --device cpu
```

## Output

The output is written in CAID format:

```text
>sequence_id
1    M    0.892000    1
2    E    0.813000    1
...