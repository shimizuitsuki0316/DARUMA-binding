"""
This script takes a single protein's ProtT5 vector in .npy format as input and performs prediction using the DARUMA-binding model.

Example usage:
    python3 predict.py -e protein1_prott5_vector.npy -o protein1_prott5_vector.out

Arguments:
    -f, --fasta: Path to the input FASTA file (optional, used if you want to specify the target sequence in FASTA format)
    -e, --embeddings: Path to the input ProtT5 vector NumPy file (required)
    -o, --output: Path to the output prediction file (required)
    -d, --device: Device to use (GPU number, e.g., "0", "1", ...). If not specified, CPU will be used.

Output:
    The prediction probabilities will be written to output_file_path.

Steps:
    1. Load input data
    2. Load the model
    3. Run prediction
    4. Output the results
"""

import sys,os
import argparse

NUM_THREADS = 8

os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import numpy as np

torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(1)

dir = os.path.dirname(__file__)
sys.path.append(dir)

from backend.models import daruma_binding
from utils.utils import read_single_fasta


def parse_args():
    parser = argparse.ArgumentParser(description="Run prediction using ProtT5 embedding vectors.")
    parser.add_argument("-f", "--fasta", required=False, help="Input FASTA file, e.g. protein1.fasta") 
    parser.add_argument("-e", "--embeddings", required=True, help="Input ProtT5 vector file, e.g. protein1_prott5_vector.npy")
    parser.add_argument("-o", "--output", required=True, help="Output prediction file, e.g. protein1_prott5_vector.out")
    parser.add_argument("-d", "--device", default="cpu", help="Use device")
    return parser.parse_args()

args = parse_args()
input_fasta_path = args.fasta if args.fasta else None
input_embeddings_path = args.embeddings
output_file_path = args.output
target_device = args.device


if torch.cuda.is_available() and target_device != "cpu":
    torch.cuda.set_device(int(target_device))
    device = torch.device(f"cuda:{target_device}")
else:
    device = torch.device("cpu")
print(f"Using {device} device.")


# 1. Load input data
acc, seq = read_single_fasta(args.fasta) if args.fasta else (None, None)

X = np.load(input_embeddings_path)
X = torch.from_numpy(X).to(device=device, dtype=torch.float32)
print(f"Input data: {input_embeddings_path}, shape: {X.shape}")

if seq is not None and X.shape[0] != len(seq):
    raise ValueError(f"Length of sequence ({len(seq)}) does not match the first dimension of ProtT5 vector ({X.shape[0]}).")

if seq is None:
    seq = ["-"] * X.shape[0]

# 2. Load the model
model = daruma_binding(device)

# 3. Run prediction
X = X.unsqueeze(0).permute(0, 2, 1).to(device=device, dtype=torch.float32)
prob, pred = model.predict_from_x(X)

# 4. Output the results
"""
>sequence_id
1    M    0.892    1
2    E    0.813    1
...
"""
with open(output_file_path, "w") as f:
    if acc:
        f.write(f">{acc}\n")
    else:
        f.write(">sequence\n")
    for i, (aa, p) in enumerate(zip(seq, prob), start=1):
        pred = 1 if p > model.threshold else 0
        f.write(f"{i}\t{aa}\t{p:.6f}\t{pred}\n")

print(f"Prediction saved to: {output_file_path}")
