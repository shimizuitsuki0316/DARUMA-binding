import numpy as np
import sys, os
import re
from datetime import datetime
import time
from struct import unpack
import torch
from torch import nn
from transformers import T5Tokenizer, T5EncoderModel
import hashlib
import base64

dir_name = os.path.dirname(__file__)
sys.path.append(os.path.join(dir_name, ".."))
from backend.functions import load_AAvectors, load_onehotvectors


def read_dataset(path):
    dic = {}
    with open(path,"r") as f:
        data = f.read().strip(">").split(">")
        for block in data:
            ac, seq, label = block.strip().split("\n")
            dic.update({ac:{"seq":seq,"label":label}})

    return dic

def read_fasta(path):
    dic = {}
    with open(path,"r") as f:
        data = f.read().strip(">").split("\n>")
        for block in data:
            ac, seq_text = block.strip().split("\n",1)
            seq = seq_text.replace("\n","").replace(" ","") # 改行と空白を除去
            dic.update({ac:{"seq":seq}})

    return dic

# アミノ酸配列から一意なIDを生成する関数
def seq_to_id_sha256_base32(seq: str, n_chars: int = 32) -> str:
    """
    アミノ酸配列 → SHA-256 → Base32 → 先頭32文字のIDを生成する
    - 同一配列 → 同一ID
    """
    # 1) SHA-256 ハッシュ（bytes）
    digest = hashlib.sha256(seq.encode("utf-8")).digest()
    # 2) Base32 エンコード（RFC 4648, A–Z + 2–7）
    b32 = base64.b32encode(digest).decode("ascii").rstrip("=")
    # 3) 先頭 n_chars 文字（80 bit）
    return b32[:n_chars]


class ProtT5Embedder:
    def __init__(
        self,
        model_name="Rostlab/prot_t5_xl_half_uniref50-enc",
        device="cuda",
        chunk_size=2048,
        overlap=512,
        return_cpu=False,
    ):
        assert overlap < chunk_size
        assert overlap % 2 == 0, "overlap は偶数推奨"

        self.device = torch.device(device)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.trim = overlap // 2
        self.return_cpu = return_cpu

        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name, 
            # do_lower_case=False,
            # local_files_only=True, # ローカルにキャッシュされたモデルを使用する。※初回はコメントアウトしてダウンロードする必要あり
        )

        self.model = (
            T5EncoderModel.from_pretrained(
                model_name,
                # local_files_only=True, # ローカルにキャッシュされたモデルを使用する。※初回はコメントアウトしてダウンロードする必要あり
            )
            .half()
            .to(self.device)
            .eval()
        )

        if self.device.type == "cpu":
            self.model = self.model.float()

    def embed(self, seq: str) -> torch.Tensor:
        """
        seq: アミノ酸配列（str）
        return: [L, hidden_dim] の残基埋め込み
        """
        seq = re.sub(r"[UZOB]", "X", seq)
        aa = list(seq)
        L = len(aa)

        step = self.chunk_size - self.overlap
        out_chunks = []

        with torch.inference_mode():
            start = 0
            idx = 0
            while start < L:
                end = min(start + self.chunk_size, L)
                chunk = " ".join(aa[start:end])

                tok = self.tokenizer(
                    chunk,
                    add_special_tokens=True,
                    return_tensors="pt",
                    padding=False,
                )
                input_ids = tok["input_ids"].to(self.device, non_blocking=True)
                attention_mask = tok["attention_mask"].to(self.device, non_blocking=True)

                hidden = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state

                # </s> を除いた残基部分
                residue = hidden[0, : (end - start), :]  # [chunk_len, dim]

                # ===== 採用ルール =====
                # 先頭または末尾の場合は端まで採用。中央の場合は両端を trim して採用。
                if idx == 0: # 先頭チャンク
                    take_s = 0
                    take_e = (end - start) - self.trim if end < L else (end - start) # すでに末尾の場合は trim しない
                else: # 中間・末尾チャンク
                    take_s = self.trim
                    take_e = (end - start) - self.trim if end < L else (end - start)

                residue = residue[take_s:take_e]

                if self.return_cpu:
                    residue = residue.detach().cpu()
                else:
                    residue = residue.detach()

                out_chunks.append(residue)

                # メモリ即解放
                del hidden, input_ids, attention_mask, tok

                if end == L:
                    break

                start += step
                idx += 1

        emb = torch.cat(out_chunks, dim=0)

        # デバッグ時のみ有効化
        # assert emb.shape[0] == L, (emb.shape[0], L)

        return emb
    
    def __call__(self, seq: str) -> torch.Tensor:
        return self.embed(seq)

class SequenceFeaturizer:
    """
    アミノ酸配列→特徴量の変換を管理するクラス
    """
    def __init__(self, feature_type, device, zarr_path=None):
        """
        feature_type: "AAindex" or "OneHot" or "ProtT5" or "AAindex_ProtT5"
        """
        self.feature_type = feature_type
        self.device = device

        self.embedding_method = ProtT5Embedder(device=self.device, chunk_size=2048, overlap=512)

    def featurize(self, seq, save_to_zarr=False):
        """
        配列を特徴量に変換するメソッド
        """
        # ファイルに保存されたProtT5ベクトルを読み込む場合
        if self.feature_type == "AAindex":
            return self.embedding_method(seq)

        elif self.feature_type == "OneHot":
            return self.embedding_method(seq)
        
        elif self.feature_type == "ProtT5_old":
            print("Warning: Using old ProtT5 embedding method without chunking.")
            return self.embedding_method(seq)
        
        elif self.feature_type == "ProtT5_noZarr":
            return self.embedding_method(seq)
        
        elif self.feature_type == "ProtT5":
            for self.zarr_manager in self.zarr_managers: # 複数のZarrManagerを順に試す
                emb = self.zarr_manager.get_embedding(seq) # 埋め込みベクトルを取得
                if emb is not None: # 見つかった場合
                    # print(f"Loaded ProtT5 embedding from zarr for sequence_id: {seq_to_id_sha256_base32(seq)}, length: {emb.shape[0]}")
                    return torch.from_numpy(emb) # float16のnumpy配列をテンソルに変換して返す
            
            # 見つからなかった場合 
            print("Warning: Zarrに無い配列を検出。")
            emb = self.embedding_method(seq) # 埋め込みベクトルを計算
            if save_to_zarr: # 保存フラグが立っている場合
                self.zarr_manager.append_protein(seq, emb.cpu().numpy())
                print(f"Saved ProtT5 embedding to zarr for sequence_id: {seq_to_id_sha256_base32(seq)}, length: {emb.shape[0]}")
            return emb
        
        elif self.feature_type == "AAindex_ProtT5":
            # AAindex取得
            aaindex_emb = self._vectorize_method(seq) # (L, 553)
            # ProtT5取得
            protT5_emb = None
            for self.zarr_manager in self.zarr_managers: # 複数のZarrManagerを順に試す
                protT5_emb = self.zarr_manager.get_embedding(seq) # 埋め込みベクトルを取得
                if protT5_emb is not None: # 見つかった場合
                    protT5_emb = torch.from_numpy(protT5_emb) # float16のnumpy配列をテンソルに変換

            if protT5_emb is None: # 見つからなかった場合
                print("Warning: Zarrに無い配列を検出。")
                protT5_emb = self.protT5_embedder(seq).cpu() # 埋め込みベクトルを計算してCPUに移動
                if save_to_zarr: # 保存フラグが立っている場合
                    self.zarr_manager.append_protein(seq, protT5_emb.cpu().numpy())
                    print(f"Saved ProtT5 embedding to zarr for sequence_id: {seq_to_id_sha256_base32(seq)}, length: {protT5_emb.shape[0]}")

            if aaindex_emb.shape[0] != protT5_emb.shape[0]:
                raise ValueError(f"Length mismatch between AAindex and ProtT5 embeddings: {aaindex_emb.shape[0]} vs {protT5_emb.shape[0]}")
                
            # 結合して返す
            return torch.cat([aaindex_emb, protT5_emb], dim=1) # (L, 553 + ProtT5の次元数)
        
        print("error")
        return None

    def _pLM_method(self, seq):
        """
        pLM埋め込みを取得するメソッド
        """
        # 前処理
        seq = re.sub(r"[UZOB]", "X", seq)  # 希少/曖昧なアミノ酸をXに置換
        tokenized = self.tokenizer(" ".join(list(seq)), add_special_tokens=True, return_tensors="pt")
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        # エンコード
        # with torch.no_grad():
        with torch.inference_mode():
            embedding_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # 後処理
        emb = embedding_repr.last_hidden_state[0, :len(seq), :]  # パディングと特殊トークンを除去
        return emb
    
    def _vectorize_method(self, seq):
        """
        AAindexベクトルに変換するメソッド
        """
        res_indices = torch.tensor([self.aa_to_index[res] for res in seq])
        return self.embedding_layer(res_indices)
    
class SeqdataManerger:
    def __init__(self, path, label_map, featurizer, has_ac=False):
        """
        データを管理するクラス
        """
        self.featurizer = featurizer
        self.dic = read_dataset(path)
        self.has_ac = has_ac
        
        # ラベルが「','で区切られた小数値」か「整数値または文字」かを判定
        sample_label = next(iter(self.dic.values()))["label"]
        if ',' in sample_label and all(re.match(r'^-?\d+(\.\d+)?$', l) for l in sample_label.split(',')): # もしカンマで区切られた小数値ならば
            self.label_type = "soft"
        else:
            self.label_type = "hard"

        self.aa_to_index, embedding_vectors = load_AAvectors()
        self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(embedding_vectors))
        
        self.seqs, self.labels = self._get_data(label_map)  # データセットを取得
        self.now_index = 0
        self.N = len(self.seqs)

    # def _seq_to_vectors(self,seq):
    #     res_indices = torch.tensor([self.aa_to_index[res] for res in seq]) # 単語インデックスのテンソー
    #     return self.embedding_layer(res_indices) # 単語インデックスからベクトルへ
    
    # def _seq_to_vectors_72padding(self,seq):
    #     seq = ["x"]*72 + list(seq) + ["x"]*72  # パディング
    #     res_indices = torch.tensor([self.aa_to_index[res] for res in seq]) # 単語インデックスのテンソー
    #     return self.embedding_layer(res_indices) # 単語インデックスからベクトルへ

    def _get_data(self, label_map):
        """
        データセットを取得するメソッド
        Returns:
            seqs: 配列のリスト
            labels: ラベルのリスト
        """
        seqs, labels = [], []
        if self.has_ac:
            acs = []

        for i,ac in enumerate(self.dic.keys()):
            if i % 100 == 0:
                print(f"Processing {i}th protein: {ac}")
            seq = self.dic[ac]["seq"].replace("U","X").replace("Z","X").replace("B","X")  # U,Z,BをXに変換

            if self.label_type == "soft":
                label = [float(l) for l in self.dic[ac]["label"].split(",")]
            elif self.label_type == "hard":
                if "," in self.dic[ac]["label"]:
                    label = [label_map[l] for l in self.dic[ac]["label"].split(",")]
                else:
                    label = [label_map[l] for l in self.dic[ac]["label"]]
                assert all(l in label_map.values() for l in label), f"Invalid label in {ac}: {set(label) - set(label_map.values())}"
            
            assert len(seq) == len(label), f"Length mismatch in {ac}: seq length {len(seq)}, label length {len(label)}"
            # 長さLの配列, 各要素は0,1,2のいずれか
            # embed_vector = self._seq_to_vectors(seq) # (L, 553)のテンソル

            seqs.append(seq)
            labels.append(label)
            if self.has_ac:
                acs.append(ac)
        
        if self.has_ac: # ACを管理する場合
            self.acs = acs
        
        return seqs, labels  # seqsは配列のリスト, labelsはラベルのリスト
        
    
    def batch_shuffle_reset(self):
        """
        データセットをシャッフルするメソッド
        """
        self.batch_indices = np.random.permutation(np.arange(self.N))
        self.now_index = 0

    def batch_reset(self):
        """
        実験: シャッフルしない
        """
        self.batch_indices = np.arange(self.N)
        self.now_index = 0
    
    def get_next_batch(self):
        """
        次のバッチを取得するメソッド
        """
        if self.now_index >= len(self.batch_indices): # 最後まで到達した場合
            if self.has_ac:
                return None, None, None, None
            return None, None

        indices = self.batch_indices[self.now_index]
        self.now_index += 1

        seq = self.seqs[indices]
        label = self.labels[indices]

        X = self.featurizer.featurize(seq).unsqueeze(0).permute(0, 2, 1).to(torch.float32)  # (1, D, L)
        # X = self._seq_to_vectors(seq).unsqueeze(0).permute(0, 2, 1).to(torch.float32)  # (1, 553, L)
        # X = self._seq_to_vectors_72padding(seq).unsqueeze(0).permute(0, 2, 1).to(torch.float32)  # (1, 553, L)
        if self.label_type == "soft":
            y = torch.tensor(label, dtype=torch.float32)
        else:
            y = torch.tensor(label, dtype=torch.long)

        if self.has_ac: # ACも返す場合
            ac = self.acs[indices]
            return X, y, ac, seq

        return X, y  # Xは(1, 553, L), yは(L)
    
    def get_label_counts(self):
        """
        ラベルの出現回数を取得するメソッド
        Returns:
            counts: ラベルの出現回数の辞書
        """
        if self.label_type == "soft":
            return None  # ソフトラベルの場合はNoneを返す
        
        counts = {}
        for label_seq in self.labels:
            for label in label_seq:
                if label not in counts:
                    counts[label] = 0
                counts[label] += 1
        return counts

import time
class LapTimer:
    """
    時間計測クラス
    """
    def __init__(self):
        self.start_time = None
        self.checkpoint = None
    
    def start(self):
        self.start_time = time.time()
        self.checkpoint = self.start_time
    
    def lap(self):
        lap_time = time.time() - self.checkpoint
        self.checkpoint = time.time()
        return lap_time
    
    def elapsed(self):
        return time.time() - self.start_time



class ResultWriter:
    def __init__(self,output_path,threthold=0.26):
        self.path = output_path
        with open(self.path,"w") as f:
            f.write(f"# Line1. accession\n")
            f.write(f"# Line2. sequence\n")
            f.write(f"# Line3. binding probability (float)\n")
            f.write(f"# Line4. binding prediction (1:binding, 0:non-binding), threshold={threthold}\n")
            f.write(f"# Line5. pLDDT (float)\n")
            f.write(f"# Line6. RSA (float)\n")
            f.write(f"# Line7. SS3 (H:Helix, S:Sheet, C:Coil)\n")
            f.write(f"# Line8. SS8 (H:Alpha helix, G:3-10 helix, I:pi helix, E:Extended strand, B:Isolated beta bridge, T:Turn, S:Bend, -:Other (coil))\n")

    def append_write(self,ac,result,threthold=0.26):
        """
        予測結果をファイルに書き込むメソッド
        """
        with open(self.path,"a") as f:
            f.write(f">{ac}\n")
            f.write(f"{result['seq']}\n")
            f.write(f"{','.join([f'{x:.4f}' for x in result['binding']])}\n")
            f.write(f"{','.join([str(x) for x in result['binary_binding']])}\n")
            f.write(f"{','.join([f'{x:.4f}' for x in result['plddt']])}\n")
            f.write(f"{','.join([f'{x:.4f}' for x in result['rsa']])}\n")
            f.write(f"{','.join(result['ss3_str'])}\n")
            f.write(f"{','.join(result['ss8_str'])}\n")


import re

def sanitize_filename(input_text):
    # ファイル名に使用できない文字を置換する
    safe_filename = re.sub(r'[\\/*?:"<>|]', ' ', input_text)
    # 必要に応じて空白を置換する（例：空白をアンダースコアに置換）
    safe_filename = safe_filename.replace(' ', '_')
    return safe_filename

class WebResultWriter:
    def __init__(self,output_dir):
        if not output_dir.endswith("/"):
            output_dir += "/"
        self.dir_path = output_dir
        # output_dirが存在しない場合は作成
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.dir_path = output_dir
        self.result_path = output_dir + "result/"
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        self.ac_list_path = os.path.join(self.dir_path,"ACs")
        with open(self.ac_list_path,"w") as f:
            f.write("")

    def append_write(self,ac,result,threthold=0.26):
        """
        予測結果をファイルに書き込むメソッド
        """
        ac = sanitize_filename(ac.strip())

        file_path = os.path.join(self.result_path,f"{ac}.daruma_binding")
        with open(file_path,"w") as f:
            f.write(f"# Line1. accession\n")
            f.write(f"# Line2. sequence\n")
            f.write(f"# Line3. binding probability (float)\n")
            f.write(f"# Line4. binding prediction (1:binding, 0:non-binding), threshold={threthold}\n")
            f.write(f"# Line5. pLDDT (float)\n")
            f.write(f"# Line6. RSA (float)\n")
            f.write(f"# Line7. SS3 (H:Helix, S:Sheet, C:Coil)\n")
            f.write(f"# Line8. SS8 (H:Alpha helix, G:3-10 helix, I:pi helix, E:Extended strand, B:Isolated beta bridge, T:Turn, S:Bend, -:Other (coil))\n")

            f.write(f">{ac}\n")
            f.write(f"{result['seq']}\n")
            f.write(f"{','.join([f'{x:.4f}' for x in result['binding']])}\n")
            f.write(f"{','.join([str(x) for x in result['binary_binding']])}\n")
            f.write(f"{','.join([f'{x:.4f}' for x in result['plddt']])}\n")
            f.write(f"{','.join([f'{x:.4f}' for x in result['rsa']])}\n")
            f.write(f"{','.join(result['ss3_str'])}\n")
            f.write(f"{','.join(result['ss8_str'])}\n")

        with open(self.ac_list_path,"a") as f:
            f.write(f"{ac}\n")


if __name__ == "__main__":
    device = torch.device("cuda:0")

    # テストコード
    # Featurizer = SequenceFeaturizer(feature_type="ProtT5", device=torch.device("cpu"))
    Featurizer = SequenceFeaturizer(feature_type="ProtT5", device=device)
    # seq = "PRTEINO"
    # seq = "MESLVLGVNEKTHVQLSLPVLQVRDVLVRGFGDSVEEALSEAREHLKNGTCGLVELEKGVLPQLEQPYVFIKRSDALSTNHGHKVVELVAEMDGIQYGRSGITLGVLVPHVGETPIAYRNVLLRKNGNKGAGGHSYGIDLKSYDLGDELGTDPIEDYEQNWNTKHGSGALRELTRELNGGAVTRYVDNNFCGPDGYPLDCIKDFLARAGKSMCTLSEQLDYIESKRGVYCCRDHEHEIAWFTERSDKSYEHQTPFEIKSAKKFDTFKGECPKFVFPLNSKVKVIQPRVEKKKTEGFMGRIRSVYPVASPQECNNMHLSTLMKCNHCDEVSWQTCDFLKATCEHCGTENLVIEGPTTCGYLPTNAVVKMPCPACQDPEIGPEHSVADYHNHSNIETRLRKGGRTRCFGGCVFAYVGCYNKRAYWVPRASADIGSGHTGITGDNVETLNEDLLEILSRERVNINIVGDFHLNEEVAIILASFSASTSAFIDTIKSLDYKSFKTIVESCGNYKVTKGKPVKGAWNIGQQRSVLTPLCGFPSQAAGVIRSIFARTLDAANHSIPDLQRAAVTILDGISEQSLRLVDAMVYTSDLLTNSVIIMAYVTGGLVQQTSQWLSNLLGTTVEKLRPIFEWIEAKLSAGVEFLKDAWEILKFLITGVFDIVKGQIQVASDNIKDCVKCFIDVVNKALEMCIDQVTIAGAKLRSLNLGEVFIAQSKGLYRQCIRGKEQLQLLMPLKAPKEVTFLEGDSHDTVLTSEEVVLKNGELEALETPVDSFTNGAIVGTPVCVNGLMLLEIKDKEQYCALSPGLLATNNVFRLKGGAPIKGVTFGEDTVWEVQGYKNVRITFELDERVDKVLNEKCSVYTVESGTEVTEFACVVAEAVVKTLQPVSDLLTNMGIDLDEWSVATFYLFDDAGEENFSSRMYCSFYPPDEEEEDDAECEEEEIDETCEHEYGTEDDYQGLPLEFGASAETVRVEEEEEEDWLDDTTEQSEIEPEPEPTPEEPVNQFTGYLKLTDNVAIKCVDIVKEAQSANPMVIVNAANIHLKHGGGVAGALNKATNGAMQKESDDYIKLNGPLTVGGSCLLSGHNLAKKCLHVVGPNLNAGEDIQLLKAAYENFNSQDILLAPLLSAGIFGAKPLQSLQVCVQTVRTQVYIAVNDKALYEQVVMDYLDNLKPRVEAPKQEEPPNTEDSKTEEKSVVQKPVDVKPKIKACIDEVTTTLEETKFLTNKLLLFADINGKLYHDSQNMLRGEDMSFLEKDAPYMVGDVITSGDITCVVIPSKKAGGTTEMLSRALKKVPVDEYITTYPGQGCAGYTLEEAKTALKKCKSAFYVLPSEAPNAKEEILGTVSWNLREMLAHAEETRKLMPICMDVRAIMATIQRKYKGIKIQEGIVDYGVRFFFYTSKEPVASIITKLNSLNEPLVTMPIGYVTHGFNLEEAARCMRSLKAPAVVSVSSPDAVTTYNGYLTSSSKTSEEHFVETVSLAGSYRDWSYSGQRTELGVEFLKRGDKIVYHTLESPVEFHLDGEVLSLDKLKSLLSLREVKTIKVFTTVDNTNLHTQLVDMSMTYGQQFGPTYLDGADVTKIKPHVNHEGKTFFVLPSDDTLRSEAFEYYHTLDESFLGRYMSALNHTKKWKFPQVGGLTSIKWADNNCYLSSVLLALQQLEVKFNAPALQEAYYRARAGDAANFCALILAYSNKTVGELGDVRETMTHLLQHANLESAKRVLNVVCKHCGQKTTTLTGVEAVMYMGTLSYDNLKTGVSIPCVCGRDATQYLVQQESSFVMMSAPPAEYKLQQGTFLCANEYTGNYQCGHYTHITAKETLYRIDGAHLTKMSEYKGPVTDVFYKETSYTTTIKPVSYKLDGVTYTEIEPKLDGYYKKDNAYYTEQPIDLVPTQPLPNASFDNFKLTCSNTKFADDLNQMTGFTKPASRELSVTFFPDLNGDVVAIDYRHYSASFKKGAKLLHKPIVWHINQATTKTTFKPNTWCLRCLWSTKPVDTSNSFEVLAVEDTQGMDNLACESQQPTSEEVVENPTIQKEVIECDVKTTEVVGNVILKPSDEGVKVTQELGHEDLMAAYVENTSITIKKPNELSLALGLKTIATHGIAAINSVPWSKILAYVKPFLGQAAITTSNCAKRLAQRVFNNYMPYVFTLLFQLCTFTKSTNSRIRASLPTTIAKNSVKSVAKLCLDAGINYVKSPKFSKLFTIAMWLLLLSICLGSLICVTAAFGVLLSNFGAPSYCNGVRELYLNSSNVTTMDFCEGSFPCSICLSGLDSLDSYPALETIQVTISSYKLDLTILGLAAEWVLAYMLFTKFFYLLGLSAIMQVFFGYFASHFISNSWLMWFIISIVQMAPVSAMVRMYIFFASFYYIWKSYVHIMDGCTSSTCMMCYKRNRATRVECTTIVNGMKRSFYVYANGGRGFCKTHNWNCLNCDTFCTGSTFISDEVARDLSLQFKRPINPTDQSSYIVDSVAVKNGALHLYFDKAGQKTYERHPLSHFVNLDNLRANNTKGSLPINVIVFDGKSKCDESASKSASVYYSQLMCQPILLLDQALVSDVGDSTEVSVKMFDAYVDTFSATFSVPMEKLKALVATAHSELAKGVALDGVLSTFVSAARQGVVDTDVDTKDVIECLKLSHHSDLEVTGDSCNNFMLTYNKVENMTPRDLGACIDCNARHINAQVAKSHNVSLIWNVKDYMSLSEQLRKQIRSAAKKNNIPFRLTCATTRQVVNVITTKISLKGGKIVSTCFKLMLKATLLCVLAALVCYIVMPVHTLSIHDGYTNEIIGYKAIQDGVTRDIISTDDCFANKHAGFDAWFSQRGGSYKNDKSCPVVAAIITREIGFIVPGLPGTVLRAINGDFLHFLPRVFSAVGNICYTPSKLIEYSDFATSACVLAAECTIFKDAMGKPVPYCYDTNLLEGSISYSELRPDTRYVLMDGSIIQFPNTYLEGSVRVVTTFDAEYCRHGTCERSEVGICLSTSGRWVLNNEHYRALSGVFCGVDAMNLIANIFTPLVQPVGALDVSASVVAGGIIAILVTCAAYYFMKFRRVFGEYNHVVAANALLFLMSFTILCLVPAYSFLPGVYSVFYLYLTFYFTNDVSFLAHLQWFAMFSPIVPFWITAIYVFCISLKHCHWFFNNYLRKRVMFNGVTFSTFEEAALCTFLLNKEMYLKLRSETLLPLTQYNRYLALYNKYKYFSGALDTTSYREAACCHLAKALNDFSNSGADVLYQPPQTSITSAVLQSGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDTVYCPRHVICTAEDMLNPNYEDLLIRKSNHSFLVQAGNVQLRVIGHSMQNCLLRLKVDTSNPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNHTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGKFYGPFVDRQTAQAAGTDTTITLNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCAALKELLQNGMNGRTILGSTILEDEFTPFDVVRQCSGVTFQGKFKKIVKGTHHWMLLTFLTSLLILVQSTQWSLFFFVYENAFLPFTLGIMAIAACAMLLVKHKHAFLCLFLLPSLATVAYFNMVYMPASWVMRIMTWLELADTSLSGYRLKDCVMYASALVLLILMTARTVYDDAARRVWTLMNVITLVYKVYYGNALDQAISMWALVISVTSNYSGVVTTIMFLARAIVFVCVEYYPLLFITGNTLQCIMLVYCFLGYCCCCYFGLFCLLNRYFRLTLGVYDYLVSTQEFRYMNSQGLLPPKSSIDAFKLNIKLLGIGGKPCIKVATVQSKMSDVKCTSVVLLSVLQQLRVESSSKLWAQCVQLHNDILLAKDTTEAFEKMVSLLSVLLSMQGAVDINRLCEEMLDNRATLQAIASEFSSLPSYAAYATAQEAYEQAVANGDSEVVLKKLKKSLNVAKSEFDRDAAMQRKLEKMADQAMTQMYKQARSEDKRAKVTSAMQTMLFTMLRKLDNDALNNIINNARDGCVPLNIIPLTTAAKLMVVVPDYGTYKNTCDGNTFTYASALWEIQQVVDADSKIVQLSEINMDNSPNLAWPLIVTALRANSAVKLQNNELSPVALRQMSCAAGTTQTACTDDNALAYYNNSKGGRFVLALLSDHQDLKWARFPKSDGTGTIYTELEPPCRFVTDTPKGPKVKYLYFIKGLNNLNRGMVLGSLAATVRLQAGNATEVPANSTVLSFCAFAVDPAKAYKDYLASGGQPITNCVKMLCTHTGTGQAITVTPEANMDQESFGGASCCLYCRCHIDHPNPKGFCDLKGKYVQIPTTCANDPVGFTLRNTVCTVCGMWKGYGCSCDQLREPLMQSADASTFLNRVCGVSAARLTPCGTGTSTDVVYRAFDIYNEKVAGFAKFLKTNCCRFQEKDEEGNLLDSYFVVKRHTMSNYQHEETIYNLVKDCPAVAVHDFFKFRVDGDMVPHISRQRLTKYTMADLVYALRHFDEGNCDTLKEILVTYNCCDDDYFNKKDWYDFVENPDILRVYANLGERVRQSLLKTVQFCDAMRDAGIVGVLTLDNQDLNGNWYDFGDFVQVAPGCGVPIVDSYYSLLMPILTLTRALAAESHMDADLAKPLIKWDLLKYDFTEERLCLFDRYFKYWDQTYHPNCINCLDDRCILHCANFNVLFSTVFPPTSFGPLVRKIFVDGVPFVVSTGYHFRELGVVHNQDVNLHSSRLSFKELLVYAADPAMHAASGNLLLDKRTTCFSVAALTNNVAFQTVKPGNFNKDFYDFAVSKGFFKEGSSVELKHFFFAQDGNAAISDYDYYRYNLPTMCDIRQLLFVVEVVDKYFDCYDGGCINANQVIVNNLDKSAGFPFNKWGKARLYYDSMSYEDQDALFAYTKRNVIPTITQMNLKYAISAKNRARTVAGVSICSTMTNRQFHQKLLKSIAATRGATVVIGTSKFYGGWHNMLKTVYSDVETPHLMGWDYPKCDRAMPNMLRIMASLVLARKHNTCCNLSHRFYRLANECAQVLSEMVMCGGSLYVKPGGTSSGDATTAYANSVFNICQAVTANVNALLSTDGNKIADKYVRNLQHRLYECLYRNRDVDHEFVDEFYAYLRKHFSMMILSDDAVVCYNSNYAAQGLVASIKNFKAVLYYQNNVFMSEAKCWTETDLTKGPHEFCSQHTMLVKQGDDYVYLPYPDPSRILGAGCFVDDIVKTDGTLMIERFVSLAIDAYPLTKHPNQEYADVFHLYLQYIRKLHDELTGHMLDMYSVMLTNDNTSRYWEPEFYEAMYTPHTVLQAVGACVLCNSQTSLRCGACIRRPFLCCKCCYDHVISTSHKLVLSVNPYVCNAPGCDVTDVTQLYLGGMSYYCKSHKPPISFPLCANGQVFGLYKNTCVGSDNVTDFNAIATCDWTNAGDYILANTCTERLKLFAAETLKATEETFKLSYGIATVREVLSDRELHLSWEVGKPRPPLNRNYVFTGYRVTKNSKVQIGEYTFEKGDYGDAVVYRGTTTYKLNVGDYFVLTSHTVMPLSAPTLVPQEHYVRITGLYPTLNISDEFSSNVANYQKVGMQKYSTLQGPPGTGKSHFAIGLALYYPSARIVYTACSHAAVDALCEKALKYLPIDKCSRIIPARARVECFDKFKVNSTLEQYVFCTVNALPETTADIVVFDEISMATNYDLSVVNARLRAKHYVYIGDPAQLPAPRTLLTKGTLEPEYFNSVCRLMKTIGPDMFLGTCRRCPAEIVDTVSALVYDNKLKAHKDKSAQCFKMFYKGVITHDVSSAINRPQIGVVREFLTRNPAWRKAVFISPYNSQNAVASKILGLPTQTVDSSQGSEYDYVIFTQTTETAHSCNVNRFNVAITRAKIGILCIMSDRDLYDKLQFTSLEIPRRNVATLQAENVTGLFKDCSKIITGLHPTQAPTHLSVDIKFKTEGLCVDIPGIPKDMTYRRLISMMGFKMNYQVNGYPNMFITREEAIRHVRAWIGFDVEGCHATRDAVGTNLPLQLGFSTGVNLVAVPTGYVDTENNTEFTRVNAKPPPGDQFKHLIPLMYKGLPWNVVRIKIVQMLSDTLKGLSDRVVFVLWAHGFELTSMKYFVKIGPERTCCLCDKRATCFSTSSDTYACWNHSVGFDYVYNPFMIDVQQWGFTGNLQSNHDQHCQVHGNAHVASCDAIMTRCLAVHECFVKRVDWSVEYPIIGDELRVNSACRKVQHMVVKSALLADKFPVLHDIGNPKAIKCVPQAEVEWKFYDAQPCSDKAYKIEELFYSYATHHDKFTDGVCLFWNCNVDRYPANAIVCRFDTRVLSNLNLPGCDGGSLYVNKHAFHTPAFDKSAFTNLKQLPFFYYSDSPCESHGKQVVSDIDYVPLKSATCITRCNLGGAVCRHHANEYRQYLDAYNMMISAGFSLWIYKQFDTYNLWNTFTRLQSLENVAYNVVNKGHFDGHAGEAPVSIINNAVYTKVDGIDVEIFENKTTLPVNVAFELWAKRNIKPVPEIKILNNLGVDIAANTVIWDYKREAPAHVSTIGVCTMTDIAKKPTESACSSLTVLFDGRVEGQVDLFRNARNGVLITEGSVKGLTPSKGPAQASVNGVTLIGESVKTQFNYFKKVDGIIQQLPETYFTQSRDLEDFKPRSQMETDFLELAMDEFIQRYKLEGYAFEHIVYGDFSHGQLGGLHLMIGLAKRSQDSPLKLEDFIPMDSTVKNYFITDAQTGSSKCVCSVIDLLLDDFVEIIKSQDLSVISKVVKVTIDYAEISFMLWCKDGHVETFYPKLQASQAWQPGVAMPNLYKMQRMLLEKCDLQNYGENAVIPKGIMMNVAKYTQLCQYLNTLTLAVPYNMRVIHFGAGSDKGVAPGTAVLRQWLPTGTLLVDSDLNDFVSDADSTLIGDCATVHTANKWDLIISDMYDPRTKHVTKENDSKEGFFTYLCGFIKQKLALGGSIAVKITEHSWNADLYKLMGHFSWWTAFVTNVNASSSEAFLIGANYLGKPKEQIDGYTMHANYIFWRNTNPIQLSSYSLFDMSKFPLKLRGTAVMSLKENQINDMIYSLLEKGRLIIRENNRVVVSSDILVNN"
    # seq = "MESLVLGVNEKTHVQLSLPVLQVRDVLVRGFGDSVEEALSEAREHLKNGTCGLVELEKGVLPQLEQPYVFIKRSDALSTNHGHKVVELVAEMDGIQYGRSGITLGVLVPHVGETPIAYRNVLLRKNGNKGAGGHSYGIDLKSYDLGDELGTDPIEDYEQNWNTKHGSGALRELTRELNGGAVTRYVDNNFCGPDGYPLDCIKDFLARAGKSMCTLSEQLDYIESKRGVYCCRDHEHEIAWFTERSDKSYEHQTPFEIKSAKKFDTFKGECPKFVFPLNSKVKVIQPRVEKKKTEGFMGRIRSVYPVASPQECNNMHLSTLMKCNHCDEVSWQTCDFLKATCEHCGTENLVIEGPTTCGYLPTNAVVKMPCPACQDPEIGPEHSVADYHNHSNIETRLRKGGRTRCFGGCVFAYVGCYNKRAYWVPRASADIGSGHTGITGDNVETLNEDLLEILSRERVNINIVGDFHLNEEVAIILASFSASTSAFIDTIKSLDYKSFKTIVESCGNYKVTKGKPVKGAWNIGQQRSVLTPLCGFPSQAAGVIRSIFARTLDAANHSIPDLQRAAVTILDGISEQSLRLVDAMVYTSDLLTNSVIIMAYVTGGLVQQTSQWLSNLLGTTVEKLRPIFEWIEAKLSAGVEFLKDAWEILKFLITGVFDIVKGQIQVASDNIKDCVKCFIDVVNKALEMCIDQVTIAGAKLRSLNLGEVFIAQSKGLYRQCIRGKEQLQLLMPLKAPKEVTFLEGDSHDTVLTSEEVVLKNGELEALETPVDSFTNGAIVGTPVCVNGLMLLEIKDKEQYCALSPGLLATNNVFRLKGGAPIKGVTFGEDTVWEVQGYKNVRITFELDERVDKVLNEKCSVYTVESGTEVTEFACVVAEAVVKTLQPVSDLLTNMGIDLDEWSVATFYLFDDAGEENFSSRMYCSFYPPDEEEEDDAECEEEEIDETCEHEYGTEDDYQGLPLEFGASAETVRVEEEEEEDWLDDTTEQSEIEPEPEPTPEEPVNQFTGYLKLTDNVAIKCVDIVKEAQSANPMVIVNAANIHLKHGGGVAGALNKATNGAMQKESDDYIKLNGPLTVGGSCLLSGHNLAKKCLHVVGPNLNAGEDIQLLKAAYENFNSQDILLAPLLSAGIFGAKPLQSLQVCVQTVRTQVYIAVNDKALYEQVVMDYLDNLKPRVEAPKQEEPPNTEDSKTEEKSVVQKPVDVKPKIKACIDEVTTTLEETKFLTNKLLLFADINGKLYHDSQNMLRGEDMSFLEKDAPYMVGDVITSGDITCVVIPSKKAGGTTEMLSRALKKVPVDEYITTYPGQGCAGYTLEEAKTALKKCKSAFYVLPSEAPNAKEEILGTVSWNLREMLAHAEETRKLMPICMDVRAIMATIQRKYKGIKIQEGIVDYGVRFFFYTSKEPVASIITKLNSLNEPLVTMPIGYVTHGFNLEEAARCMRSLKAPAVVSVSSPDAVTTYNGYLTSSSKTSEEHFVETVSLAGSYRDWSYSGQRTELGVEFLKRGDKIVYHTLESPVEFHLDGEVLSLDKLKSLLSLREVKTIKVFTTVDNTNLHTQLVDMSMTYGQQFGPTYLDGADVTKIKPHVNHEGKTFFVLPSDDTLRSEAFEYYHTLDESFLGRYMSALNHTKKWKFPQVGGLTSIKWADNNCYLSSVLLALQQLEVKFNAPALQEAYYRARAGDAANFCALILAYSNKTVGELGDVRETMTHLLQHANLESAKRVLNVVCKHCGQKTTTLTGVEAVMYMGTLSYDNLKTGVSIPCVCGRDATQYLVQQESSFVMMSAPPAEYKLQQGTFLCANEYTGNYQCGHYTHITAKETLYRIDGAHLTKMSEYKGPVTDVFYKETSYTTTIKPVSYKLDGVTYTEIEPKLDGYYKKDNAYYTEQPIDLVPTQPLPNASFDNFKLTCSNTKFADDLNQMTGFTKPASRELSVTFFPDLNGDVVAIDYRHYSASFKKGAKLLHKPIVWHINQATTKTTFKPNTWCLRCLWSTKPVDTSNSFEVLAVEDTQGMDNLACESQQPTSEEVVENPTIQKEVIECDVKTTEVVGNVILKPSDEGVKVTQELGHEDLMAAYVENTSITIKKPNELSLALGLKTIATHGIAAINSVPWSKILAYVKPFLGQAAITTSNCAKRLAQRVFNNYMPYVFTLLFQLCTFTKSTNSRIRASLPTTIAKNSVKSVAKLCLDAGINYVKSPKFSKLFTIAMWLLLLSICLGSLICVTAAFGVLLSNFGAPSYCNGVRELYLNSSNVTTMDFCEGSFPCSICLSGLDSLDSYPALETIQVTISSYKLDLTILGLAAEWVLAYMLFTKFFYLLGLSAIMQVFFGYFASHFISNSWLMWFIISIVQMAPVSAMVRMYIFFASFYYIWKSYVHIMDGCTSSTCMMCYKRNRATRVECTTIVNGMKRSFYVYANGGRGFCKTHNWNCLNCDTFCTGSTFISDEVARDLSLQFKRPINPTDQSSYIVDSVAVKNGALHLYFDKAGQKTYERHPLSHFVNLDNLRANNTKGSLPINVIVFDGKSKCDESASKSASVYYSQLMCQPILLLDQALVSDVGDSTEVSVKMFDAYVDTFSATFSVPMEKLKALVATAHSELAKGVALDGVLSTFVSAARQGVVDTDVDTKDVIECLKLSHHSDLEVTGDSCNNFMLTYNKVENMTPRDLGACIDCNARHINAQVAKSHNVSLIWNVKDYMSLSEQLRKQIRSAAKKNNIPFRLTCATTRQVVNVITTKISLKGGKIVSTCFKLMLKATLLCVLAALVCYIVMPVHTLSIHDGYTNEIIGYKAIQDGVTRDIISTDDCFANKHAGFDAWFSQRGGSYKNDKSCPVVAAIITREIGFIVPGLPGTVLRAINGDFLHFLPRVFSAVGNICYTPSKLIEYSDFATSACVLAAECTIFKDAMGKPVPYCYDTNLLEGSISYSELRPDTRYVLMDGSIIQFPNTYLEGSVRVVTTFDAEYCRHGTCERSEVGICLSTSGRWVLNNEHYRALSGVFCGVDAMNLIANIFTPLVQPVGALDVSASVVAGGIIAILVTCAAYYFMKFRRVFGEYNHVVAANALLFLMSFTILCLVPAYSFLPGVYSVFYLYLTFYFTNDVSFLAHLQWFAMFSPIVPFWITAIYVFCISLKHCHWFFNNYLRKRVMFNGVTFSTFEEAALCTFLLNKEMYLKLRSETLLPLTQYNRYLALYNKYKYFSGALDTTSYREAACCHLAKALNDFSNSGADVLYQPPQTSITSAVLQSGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDTVYCPRHVICTAEDMLNPNYEDLLIRKSNHSFLVQAGNVQLRVIGHSMQNCLLRLKVDTSNPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNHTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGKFYGPFVDRQTAQAAGTDTTITLNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCAALKELLQNGMNGRTILGSTILEDEFTPFDVVRQCSGVTFQGKFKKIVKGTHHWMLLTFLTSLLILVQSTQWSLFFFVYENAFLPFTLGIMAIAACAMLLVKHKHAFLCLFLLPSLATVAYFNMVYMPASWVMRIMTWLELADTSLSGYRLKDCVMYASALVLLILMTARTVYDDAARRVWTLMNVITLVYKVYYGNALDQAISMWALVISVTSNYSGVVTTIMFLARAIVFVCVEYYPLLFITGNTLQCIMLVYCFLGYCCCCYFGLFCLLNRYFRLTLGVYDYLVSTQEFRYMNSQGLLPPKSSIDAFKLNIKLLGIGGKPCIKVATVQSKMSDVKCTSVVLLSVLQQLRVESSSKLWAQCVQLHNDILLAKDTTEAFEKMVSLLSVLLSMQGAVDINRLCEEMLDNRATLQAIASEFSSLPSYAAYATAQEAYEQAVANGDSEVVLKKLKKSLNVAKSEFDRDAAMQRKLEKMADQAMTQMYKQARSEDKRAKVTSAMQTMLFTMLRKLDNDALNNIINNARDGCVPLNIIPLTTAAKLMVVVPDYGTYKNTCDGNTFTYASALWEIQQVVDADSKIVQLSEINMDNSPNLAWPLIVTALRANSAVKLQNNELSPVALRQMSCAAGTTQTACTDDNALAYYNNSKGGRFV"
    # seq = "MESLVLGVNEKTHVQLSLPVLQVRDVLVRGFGDSVEEALSEAREHLKNGTCGLVELEKGVLPQLEQPYVFIKRSDALSTNHGHKVVELVAEMDGIQYGRSGITLGVLVPHVGETPIAYRNVLLRKNGNKGAGGHSYGIDLKSYDLGDELGTDPIEDYEQNWNTKHGSGALRELTRELNGGAVTRYVDNNFCGPDGYPLDCIKDFLARAGKSMCTLSEQLDYIESKRGVYCCRDHEHEIAWFTERSDKSYEHQTPFEIKSAKKFDTFKGECPKFVFPLNSKVKVIQPRVEKKKTEGFMGRIRSVYPVASPQECNNMHLSTLMKCNHCDEVSWQTCDFLKATCEHCGTENLVIEGPTTCGYLPTNAVVKMPCPACQDPEIGPEHSVADYHNHSNIETRLRKGGRTRCFGGCVFAYVGCYNKRAYWVPRASADIGSGHTGITGDNVETLNEDLLEILSRERVNINIVGDFHLNEEVAIILASFSASTSAFIDTIKSLDYKSFKTIVESCGNYKVTKGKPVKGAWNIGQQRSVLTPLCGFPSQAAGVIRSIFARTLDAANHSIPDLQRAAVTILDGISEQSLRLVDAMVYTSDLLTNSVIIMAYVTGGLVQQTSQWLSNLLGTTVEKLRPIFEWIEAKLSAGVEFLKDAWEILKFLITGVFDIVKGQIQVASDNIKDCVKCFIDVVNKALEMCIDQVTIAGAKLRSLNLGEVFIAQSKGLYRQCIRGKEQLQLLMPLKAPKEVTFLEGDSHDTVLTSEEVVLKNGELEALETPVDSFTNGAIVGTPVCVNGLMLLEIKDKEQYCALSPGLLATNNVFRLKGGAPIKGVTFGEDTVWEVQGYKNVRITFELDERVDKVLNEKCSVYTVESGTEVTEFACVVAEAVVKTLQPVSDLLTNMGIDLDEWSVATFYLFDDAGEENFSSRMYCSFYPPDEEEEDDAECEEEEIDETCEHEYGTEDDYQGLPLEFGASAETVRVEEEEEEDWLDDTTEQSEIEPEPEPTPEEPVNQFTGYLKLTDNVAIKCVDIVKEAQSANPMVIVNAANIHLKHGGGVAGALNKATNGAMQKESDDYIKLNGPLTVGGSCLLSGHNLAKKCLHVVGPNLNAGEDIQLLKAAYENFNSQDILLAPLLSAGIFGAKPLQSLQVCVQTVRTQVYIAVNDKALYEQVVMDYLDNLKPRVEAPKQEEPPNTEDSKTEEKSVVQKPVDVKPKIKACIDEVTTTLEETKFLTNKLLLFADINGKLYHDSQNMLRGEDMSFLEKDAPYMVGDVITSGDITCVVIPSKKAGGTTEMLSRALKKVPVDEYITTYPGQGCAGYTLEEAKTALKKCKSAFYVLPSEAPNAKEEILGTVSWNLREMLAHAEETRKLMPICMDVRAIMATIQRKYKGIKIQEGIVDYGVRFFFYTSKEPVASIITKLNSLNEPLVTMPIGYVTHGFNLEEAARCMRSLKAPAVVSVSSPDAVTTYNGYLTSSSKTSEEHFVETVSLAGSYRDWSYSGQRTELGVEFLKRGDKIVYHTLESPVEFHLDGEVLSLDKLKSLLSLREVKTIKVFTTVDNTNLHTQLVDMSMTYGQQFGPTYLDGADVTKIKPHVNHEGKTFFVLPSDDTLRSEAFEYYHTLDESFLGRYMSALNHTKKWKFPQVGGLTSIKWADNNCYLSSVLLALQQLEVKFNAPALQEAYYRARAGDAANFCALILAYSNKTVGELGDVRETMTHLLQHANLESAKRVLNVVCKHCGQKTTTLTGVEAVMYMGTLSYDNLKTGVSIPCVCGRDATQYLVQQESSFVMMSAPPAEYKLQQGTFLCANEYTGNYQCGHYTHITAKETLYRIDGAHLTKMSEYKGPVTDVFYKETSYTTTIKPVSYKLDGVTYTEIEPKLDGYYKKDNAYYTEQPIDLVPTQPLPNASFDNFKLTCSNTKFADDLNQMTGFTKPASRELSVTFFPDLNGDVVAIDYRHYSASFKKGAKLLHKPIVWHINQATTKTTFKPNTWCLRCLWSTKPVDTSNSFEVLAVEDTQGMDNLACESQQPTSEEVVENPTIQKEVIECDVKTTEVVGNVILKPSDEGVKVTQELGHEDLMAAYVENTSITIKKPNELSLALGLKTIATHGIAAINSVPWSKILAYVKPFLGQAAITTSNCAKRLAQRVFNNYMPYVFTLLFQLCTFTKSTNSRIRASLPTTIAKNSVKSVAKLCLDAGINYVKSPKFSKLFTIAMWLLLLSICLGSLICVTAAFGVLLSNFGAPSYCNGVRELYLNSSNVTTMDFCEGSFPCSICLSGLDSLDSYPALETIQVTISSYKLDLTILGLAAEWVLAYMLFTKFFYLLGLSAIMQVFFGYFASHFISNSWLMWFIISIVQMAPVSAMVRMYIFFASFYYIWKSYVHIMDGCTSSTCMMCYKRNRATRVECTTIVNGMKRSFYVYANGGRGFCKTHNWNCLNCDTFCTGSTFISDEVARDLSLQFKRPINPTDQSSYIVDSVAVKNGALHLYFDKAGQKTYERHPLSHFVNLDNLRANNTKGSLPINVIVFDGKSKCDESASKSASVYYSQLMCQPILLLDQALVSDVGDSTEVSVKMFDAYVDTFSATFSVPMEKLKALVATAHSELAKGVALDGVLSTFVSAARQGVVDTDVDTKDVIECLKLSHHSDLEVTGDSCNNFMLTYNKVENMTPRDLGACIDCNARHINAQVAKSHNVSLIWNVKDYMSLSEQLRKQIRSAAKKNNIPFRLTCATTRQVVNVITTKISLKGGKIVSTCFKLMLKATLLCVLAALVCYIVMPVHTLSIHDGYTNEIIGYKAIQDGVTRDIISTDDCFANKHAGFDAWFSQRGGSYKNDKSCPVVAAIITREIGFIVPGLPGTVLRAINGDFLHFLPRVFSAVGNICYTPSKLIEYSDFATSACVLAAECTIFKDAMGKPVPYCYDTNLLEGSISYSELRPDTRYVLMDGSIIQFPNTYLEGSVRVVTTFDAEYCRHGTCERSEVGICLSTSGRWVLNNEHYRALSGVFCGVDAMNLIANIFTPLVQPVGALDVSASVVAGGIIAILVTCAAYYFMKFRRVFGEYNHVVAANALLFLMSFTILCLVPAYSFLPGVYSVFYLYLTFYFTNDVSFLAHLQWFAMFSPIVPFWITAIYVFCISLKHCHWFFNNYLRKRVMFNGVTFSTFEEAALCTFLLNKEMYLKLRSETLLPLTQYNRYLALYNKYKYFSGALDTTSYREAACCHLAKALNDFSNSGADVLYQPPQTSITSAVLQSGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDTVYCPRHVICTAEDMLNPNYEDLLIRKSNHSFLVQAGNVQLRVIGHSMQNCLLRLKVDTSNPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNHTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGKFYGPFVDRQTAQAAGTDTTITLNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCAALKELLQNGMNGRTILGSTILEDEFTPFDVVRQCSGVTFQGKFKKIVKGTHHWMLLTFLTSLLILVQSTQWSLFFFVYENAFLPFTLGIMAIAACAMLLVKHKHAFLCLFLLPSLATVAYFNMVYMPASWVMRIMTWLELADTSLSGYRLKDCVMYASALVLLILMTARTVYDDAARRVWTLMNVITLVYKVYYGNALDQAISMWALVISVTSNYSGVVTTIMFLARAIVFVCVEYYPLLFITGNTLQCIMLVYCFLGYCCCCYFGLFCLLNRYFRLTLGVYDYLVSTQEFRYMNSQGLLPPKSSIDAFKLNIKLLGIGGKPCIKVATVQSKMSDVKCTSVVLLSVLQQLRVESSSKLWAQCVQLHNDILLAKDTTEAFEKMVSLLSVLLSMQGAVDINRLCEEMLDNRATLQAIASEFSSLPSYAAYATAQEAYEQAVANGDSEVVLKKLKKSLNVAKSEFDRDAAMQRKLEKMADQAMTQMYKQARSEDKRA"
    # seq = "MSLEQKKGADIISKILQIQNSIGKTTSPSTLKTKLSEISRKEQENARIQSKLSDLQKKKIDIDNKLLKEKQNLIKEEILERKKLEVLTKKQQKDEIEHQKKLKREIDAIKASTQYITDVSISSYNNTIPETEPEYDLFISHASEDKEDFVRPLAETLQQLGVNVWYDEFTLKVGDSLRQKIDSGLRNSKYGTVVLSTDFIKKDWTNYELDGLVAREMNGHKMILPIWHKITKNDVLDYSPNLADKVALNTSVNSIEEIAHQLADVILNR"
    # seq = "FVQWFSKFLGRIL1111111111111"
    # seq = "MSEYIRVTEDENDEPIEIPSEDDGTVLLSTVTAQFPGACGLRYRNPVSQCMRGVRLVEGILHAPDAGWGNLVYVVNYPKDNKRKMDETDASSAVKVKRAVQKTSDLIVLGLPWKTTEQDLKEYFSTFGEVLMVQVKKDLKTGHSKGFGFVRFTEYETQVKVMSQRHMIDGRWCDCKLPNSKQSQDEPLRSRKVFVGRCTEDMTEDELREFFSQYGDVMDVFIPKPFRAFAFVTFADDQIAQSLCGEDLIIKGISVHISNAEPKHNSNRQLERSGRFGGNPGGFGNQGGFGNSRGGGAGLGNNQGSNMGGGMNFGAFSINPAMMAAAQAALQSSWGMMGMLASQQNQSGPSGNNQNQGNMQREPNQAFGSGNNSYSGSNSGAAIGWGSASNAGSGSGFNGGFGSSMDSKSSGWGM"
    seq = "FVQWFSKFLGRILLL"
    
    emb = Featurizer.featurize(seq)
    print("seq:", seq)
    print("seq length:", len(seq))
    print(emb.shape)  # (L, 1024)
    print(emb.dtype)  # torch.float16 or torch.float32
    print(emb)  # 埋め込みベクトル

    def test_dataset_load():
        import time

        dataset_path = "/home/ishimizu/daruma2_project/ProS_predict/data/raw/caid1_disprot-binding.txt"
        # dataset_path = "/home/ishimizu/daruma2_project/ProS_predict/data/raw/caid2_binding.fasta"
        
        datamanerger = SeqdataManerger(dataset_path, label_map={"0":0, "1":1}, featurizer=Featurizer)
        datamanerger.batch_reset()

        start_time = time.time()

        while True:
            X, y = datamanerger.get_next_batch()
        
            if X is None or y is None:
                break
            
            X, y = X.to(device), y.to(device)

        end_time = time.time()
        print(f"Total time taken: {end_time - start_time:.2f} seconds")

    # test_dataset_load()

    # 確認
    def save_embedding_as_npy():
        
        # ac = "DP03914"
        # seq = "MQARREELLAKKARLAEIKRQRELRAQQAAGRSITPSELVSPTPSRANSRREIESLIDSILSSSAGANSPRRGSRPNSVISTGELSTDNASEITATTTGQLPSQPQPLSTVPLQTVYECPPSPVKEIISYSKGVQTTEEWTPRKSRGVGDSDDEDLGAAASPSKRLSRRERDREEELREKIRKEIEEELKATKELVADGVLKQSGAQNFPVRALTAEELDAVTKSVEFMDFIDRSTKVIEKALDQEYDILTDYTLQVHDVEDEDEQSGNVGGKGRRKVREIAQFYDERWSKKRMISSIDFSPKFSELLLASYTKNPTAPHDPDGIVQVWNLHLHDRPEFVFHAQSDILTAKFSPFHPNLIIGGAYSGQVLLWDTRARSAPVQKTPLTGSGHTHPVYCIDIVGTQNANNIISCSTDGAVCGWSVDMLAQPQESLTLLAPLPAKSEDLSPTCMAFPQADPTFFLVGTEEGTIYPCHRYDRAGAKAGVDPRVSYRGHAGPVMSVAFHPARGPVDLGDLVLSSSLDWSVKLWKVRAPAATSGVLSSGSVTSPSAAAAAAAAAAAVTAPLPGSTTGSLDSQVTPLLDLVREDVVYDAAWSPIKPGVFGLVDGAGTLEIWDITVETEEPVARVVPTPRKDRGALLGMKSLNKVAWEPSEGKRLATGGLDGVVSMFEVGPDLGGKEGLRNEDWGTVKKLVNRLEAGGSVDGVGM"
        ac = "Protein_L1000"
        seq = "MNYPDTLKSFALSEAQSSRWFQYQINPAQRGQNNSAFCARVKGLTAEKLEDALNHLIKRHPMLRASFGQYDGKLGYQIAENAVIKITRHNARDLSEETLQQVVRNDCWHAFDLNNPLRVYASWYQCDDQESVMVLTFDHLAVDGWSYWILLEELGKLLTEQELPPAPERSYQDYVIWQQQWLNSNNAKKQQQFWCDNLSGHLTELQLPASNKPVINQSPSATDKITVPGKLSDTLARKLLVMAEKYGNSLFAIFLAAYQILLHRYTGQDDIFIGSIMPGRSRAPWGKLVGEFVNPVALRGQINDEMTVQEHILQAAKTIRQAIENQKYPFPKILEQIKVQRHADTHPVFQTIMTFQQPRYISNLVSLWMDQSSSTTVHWGGAELRSFPYPYYAEMPVPLMVNVIEVDQQINCGFHYDTKLFDAELITQLMNNFFNLLTAMADDEQQTIDSLPLMNESERQRILYNFNATDISYPQNVLLHQLFEQQVNYTPAAIALICDNHQLTYEELNRQANKLAHAFVAAGIQADNRIGLCMERSPEMVIALLGILKAGAAYVPLDPEYPADRLRHILTDSEPALVLSHHGLQSHLPDTSVPLWIWESGEFQATITEQQEDNLSITDLGLTSRNLAYVLYTSGSTGLPKGVMNEHRGIVNRLLWAKDEYQLNPNDLVLQKTPFSFDVSVWEFFLPLLAGAQLVMARPGGHKDPLYLLEEIESRGITTLHFVPSMLQSFIHLTPAERCPSLRQILCSGEALPYSLQQQCLSHFPHSELHNLYGPTEAAIDVTSWRCGSEKYVDLVPIGHPIANTQIYILDKLGRLVPIGVTGEIHIGGIGVARGYLNRPELTAERFVHDPFSNLPDARMYKTGDLGRWLPDGSIDYLGRNDFQVKIRGNRIELEEIETRLAQCAGVQQVVVVARAYDSADTRLIAYIIPQSGTTLNIPELREQLGNSLPDYMIPGAFVMLDTFPLTLSGKLDRRALPIPDHSSVITQEYAAPQGEIEEQ"
        print("seq:", seq)
        print("seq length:", len(seq))

        emb = Featurizer.featurize(seq)
        print(emb.shape)  # (L, 1024)

        # 埋め込みベクトルをNumPy配列に変換して保存
        emb_np = emb.cpu().numpy()
        np.save(f"{ac}.npy", emb_np)

    save_embedding_as_npy()

