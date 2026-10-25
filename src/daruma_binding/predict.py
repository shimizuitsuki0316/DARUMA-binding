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
from utils.data_functions import read_fasta, LapTimer, SequenceFeaturizer, ResultWriter, WebResultWriter

def parse_args():
    parser = argparse.ArgumentParser(description="Run prediction using ProtT5 embedding vectors.")
    parser.add_argument("-i", "--input", required=False, help="Input FASTA file, e.g. protein1.fasta")
    parser.add_argument("-o", "--output", required=True, help="Output prediction file, e.g. protein1.out")
    parser.add_argument("-d", "--device", default="0", help="Use device, e.g. cpu, 0, 1, ... for GPU")
    return parser.parse_args()

args = parse_args()
input_fasta_path = args.input
output_file_path = args.output
target_device = args.device


if target_device == "cpu":
    device = torch.device("cpu")
elif torch.cuda.is_available() and target_device != "cpu":
    torch.cuda.set_device(int(target_device))
    device = torch.device(f"cuda:{target_device}")
else:
    device = torch.device("cpu")
print(f"Using {device} device.")



# Load input data
input_datas = read_fasta(input_fasta_path)

# Initialize the featurizer
featurizer = SequenceFeaturizer(feature_type="ProtT5_noZarr", device=device)

# Load the model
model = daruma_binding(device,featurizer=featurizer)

result_writer = ResultWriter(output_file_path, model.threshold)
# result_writer = WebResultWriter(output_file_path)

for ac, values in input_datas.items():
    seq = values["seq"]
    print(f"Processing {ac} ...")

    result = model.predict_from_seq(seq)

    result_writer.append_write(ac, result, model.threshold)


