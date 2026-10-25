import numpy as np
import sys, os
import re
from datetime import datetime
import time
from struct import unpack
import torch
from torch import nn



dir_name = os.path.dirname(__file__)
sys.path.append(dir_name)


def load_AAvectors():
    aa_list = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y","X","x"]
    aa_to_index = {aa:i for i,aa in enumerate(aa_list)}

    features_path = os.path.join(os.path.dirname(__file__), 'data', 'AAindex553-Normal-X0.feature')
    with open(features_path,"rb") as f:
        binary_data = f.read()
    data = unpack(">11613f",binary_data)
    vectors = np.array(data,dtype=np.float32).reshape(21,553) # 埋め込みベクトル ndarray

    # paddingベクトルを追加
    padding_vector = np.zeros((1, 553), dtype=np.float32)
    vectors = np.concatenate((vectors, padding_vector), axis=0)

    return aa_to_index, vectors

def load_onehotvectors():
    # one-hotベクトルの読み込み
    # 20種類のアミノ酸 + 未知のアミノ酸(X) + padding(x)
    # 未知のアミノ酸(X) + padding(x)は全て0ベクトルとする
    aa_list = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y","X","x"]
    aa_to_index = {aa: i for i, aa in enumerate(aa_list)}

    # 先頭20種類のみone-hot、Xとxはゼロベクトル
    vectors = np.zeros((len(aa_list), 20), dtype=np.float32)
    vectors[:20, :] = np.eye(20, dtype=np.float32)

    return aa_to_index, vectors


# FASTAファイルを読み込む関数
def read_fasta(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    
    dic = {}
    for block in data.strip(">").split("\n>"):
        ac, seq = block.strip().split("\n", 1)
        ac = ac.strip()
        seq = seq.replace("\n", "").replace(" ", "")
        seq = re.sub(r"[UBJOZ]", "X", seq) # 不明なアミノ酸をXに置換
        dic[ac] = seq    
    return dic





if __name__ == "__main__":
    # 検証用
    aa_to_index, vectors =load_onehotvectors()
    print(aa_to_index)
    print(vectors)


