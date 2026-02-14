#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
mini_ja_gpt_all_in_one.py

============================================
■ このファイルのゴール
============================================

このスクリプトは、「ミニ日本語GPT」をゼロから作って、
それを使って簡易的なチャット（会話しているっぽいインタラクション）を
行うところまでを **1ファイル** で完結させることを目的にしています。

やっていることは、大きく分けて3つです：

  1. 学習（train）
     - 日本語テキスト（小説・レポートなど）を読み込む
     - 「次の1文字」を予測するミニTransformer（GPT風）を学習する
     - 学習済みパラメータをチェックポイントとして保存する

  2. 推論（inference）
     - 保存したチェックポイントを読み込む
     - テキストの続きを自動生成する

  3. チャット風インタラクション（chat）
     - ターミナル上で「あなた: 〜」「AI: 〜」形式でやり取りする
     - 内部的には「会話ログを1つのテキストとして結合し、その続きを生成するだけ」
       だが、簡易的にチャットボットっぽく振る舞う

質の高い会話モデルではなく、
「自分で作ったミニTransformerに対して、会話のようなプロンプトを与えて遊ぶ」
ことが目的です。

============================================
■ モデルの抽象イメージ
============================================

- 入力：文字列（例：「これはテストです。」）
- 処理：
    1. 各文字を整数IDに変換
    2. IDをベクトル（埋め込み）に変換
    3. 位置情報を加えたうえで、Transformerブロックを通す
    4. 最後に「次の文字が何か」の確率分布を出す
- 出力：
    - 「次の文字が '。' である確率」「次の文字が '、' である確率」... を含むベクトル

これを、過去の文字列から1文字先を当てるタスクで学習し、
学習後は「テキストの続きを生成するマシン」として使います。

============================================
■ 実行イメージ
============================================

[学習]
    python mini_ja_gpt_all_in_one.py \\
        --mode train \\
        --data_path data/novel_ja.txt \\
        --run_name novel

[チャット]
    python mini_ja_gpt_all_in_one.py \\
        --mode chat \\
        --ckpt_path checkpoints/novel_epoch4.pt

このファイルだけで「作る」「使う」の両方ができます。
"""

# ============================================
# ■ 必要なモジュールのインポート
# ============================================

import argparse       # コマンドライン引数（--mode など）を扱う標準ライブラリ
import math           # sqrt や π など、数学関数をまとめた標準ライブラリ
import os             # ファイルパス操作、フォルダ作成など OS 関連処理
import random         # Python標準の乱数モジュール
from dataclasses import dataclass  # 設定値をまとめるクラスを簡単に書くための仕組み

import torch          # PyTorch 本体（テンソル計算・自動微分など）
import torch.nn as nn # PyTorch のニューラルネット関連モジュール (nn.Linear, nn.Module など)
from torch.utils.data import Dataset, DataLoader  # データセット・ミニバッチ処理用


# ============================================
# ■ ハイパーパラメータをまとめる Config クラス
# ============================================

"""
- 「ハイパーパラメータ」とは、手動で決める設定値（学習率、層の数など）のこと。
- これらをバラバラな変数ではなく、Config クラスに一括で入れておくことで、
  モデルにも学習ループにも同じ設定を渡しやすくなる。

ここでは dataclass を使って「設定用の箱」を定義します。
"""

@dataclass
class Config:
    # コンテキスト長（1回の入力として何文字を見るか）
    block_size: int = 128

    # ミニバッチサイズ（1ステップで何サンプル同時に処理するか）
    batch_size: int = 64

    # 埋め込み次元（1文字を何次元のベクトルで表現するか）
    d_model: int = 256

    # Multi-Head Attention のヘッド数
    num_heads: int = 4

    # Transformer ブロックの層数
    num_layers: int = 4

    # FFN（前向き全結合ネットワーク）の中で何倍の次元に広げるか
    ff_mult: int = 4

    # エポック数（学習データを何周するか）
    num_epochs: int = 20

    # 学習率（パラメータ更新の一歩の大きさ）
    lr: float = 1e-3

    # デバイス（GPUが使えれば "cuda"、無ければ "cpu"）
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 学習中にどの頻度で文章生成サンプルを出すか（ステップ数）
    sample_every: int = 500

    # テキスト生成時に、何文字ぶん新しく生成するか
    max_generate_tokens: int = 200

    # テキスト生成時の温度（1.0より大きいとランダム寄り、小さいと保守的）
    temperature: float = 1.0


# ============================================
# ■ BPE トークナイザ（簡易実装）
# ============================================

"""
ここでは、外部ライブラリを使わずに
「超シンプルな Byte Pair Encoding(BPE)」を自前実装する。

ざっくりした流れ：

  1. 学習時
     - テキストを 1文字ずつの列にする（['こ','ん','に','ち','は',…]）
     - 「隣り合う2トークンのペア」の出現頻度を数える
     - 一番よく出るペア（例: ('に','ち')）を 1つの新トークン 'にち' にマージ
     - これを何回か繰り返して、トークン語彙を増やしていく
     - 最終的なトークン集合（文字＋結合済みトークン）と「どのペアをどの順でマージしたか」を保存

  2. エンコード時
     - 入力文字列を 1文字ずつに分解
     - 保存済み「マージ手順」を、順番に適用していく
       → 'に','ち' が並んでいたら 'にち' にまとめる、など
     - 最終的なトークン列を ID 列（整数の列）に変換

  3. デコード時
     - ID → トークン → 文字列として連結

※ 実際の GPT 系の BPE よりだいぶ簡易版だけど、
   「頻出ペアをまとめてトークンにする」という本質は同じ。
"""

from typing import List, Dict, Tuple, Union, Any


class BPETokenizer:
    """
    BPETokenizer は「学習モード」と「復元モード」の2通りで初期化される。

    - 学習モード: `BPETokenizer(text: str)`
        → テキストから BPE のマージ規則とトークン語彙を学習する

    - 復元モード: `BPETokenizer(data: dict)`
        → チェックポイントに保存しておいた
           {"tokens": [...], "merges": [...]} から復元する
    """

    def __init__(self, data: Union[str, Dict[str, Any]], max_merges: int = 1000):
        """
        data:
          - str なら、生テキスト（学習時）
          - dict なら、保存済みトークナイザ情報（復元時）

        max_merges:
          - 何回 BPE マージを行うか（大きいほど語彙が増える）
        """
        self.max_merges = max_merges

        if isinstance(data, str):
            # === 学習モード ===
            self._train_from_text(data)
        else:
            # === 復元モード ===
            # data は {"tokens": [...], "merges": [...]} を期待
            self.tokens: List[str] = list(data["tokens"])
            # merges は [["に","ち"], ["こん","にち"], ...] のような形式と想定
            self.merges: List[Tuple[str, str]] = [
                (a, b) for (a, b) in data["merges"]
            ]
            self.token2id: Dict[str, int] = {
                tok: i for i, tok in enumerate(self.tokens)
            }

        self.vocab_size: int = len(self.tokens)

    # ------------------------------------------------------------------
    # 学習時に呼ばれる内部関数
    # ------------------------------------------------------------------
    def _train_from_text(self, text: str):
        """
        生テキストから BPE のマージ規則とトークン語彙を学習する。

        アルゴリズム（かなり素直な実装）：
          1. 文字列を 1文字ずつのリストにする
          2. 隣接ペアの頻度を数える
          3. 一番頻度の高いペアをマージ（1トークンにまとめる）
          4. マージ後の列に対して、再び 2. へ戻る
          5. max_merges 回 or 有効なペアが無くなるまで繰り返す
        """
        # 初期トークン列：1文字＝1トークンの世界
        tokens: List[str] = list(text)

        # merges: どのペアをどの順番でマージしたかを記録しておく
        merges: List[Tuple[str, str]] = []

        for _ in range(self.max_merges):
            # 1) 全トークン列の隣接ペア頻度をカウント
            pair_counts: Dict[Tuple[str, str], int] = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            # 2) 最頻出のペアを1つ選ぶ
            (best_pair, best_count) = max(
                pair_counts.items(), key=lambda x: x[1]
            )
            if best_count < 2:
                # 同じペアが2回以上出てこないなら、これ以上マージしてもあまり意味がないので終了
                break

            a, b = best_pair
            merges.append(best_pair)
            new_token = a + b  # 例: ("に","ち") → "にち"

            # 3) tokens 全体に対して、このペア (a,b) をまとめる
            new_tokens: List[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    # ペアを見つけたら、結合したトークンを追加
                    new_tokens.append(new_token)
                    i += 2  # 2文字分進む
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # 最終的な tokens から語彙を構築
        vocab_set = set(tokens)

        # 元テキスト中に出た全ての 1文字は必ず語彙に含める
        for ch in set(text):
            vocab_set.add(ch)

        # マージで一度でも作られたトークンも語彙に含める
        for (a, b) in merges:
            vocab_set.add(a + b)

        # 語彙リスト（ID順の並び）を決める
        tokens_list = sorted(vocab_set)
        self.tokens: List[str] = tokens_list
        self.token2id: Dict[str, int] = {tok: i for i, tok in enumerate(tokens_list)}
        self.merges: List[Tuple[str, str]] = merges

    # ------------------------------------------------------------------
    # エンコード・デコード処理
    # ------------------------------------------------------------------
    def _apply_merges(self, tokens: List[str]) -> List[str]:
        """
        学習済み merges（マージ手順）を順番に適用して、
        文字列リスト → BPEトークン列 に変換する内部関数。
        """
        for (a, b) in self.merges:
            new_tokens: List[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(a + b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str) -> List[int]:
        """
        文字列 → トークンID列 に変換。
        1. まずは文字単位に分解
        2. merges を順番に適用して BPE トークン列に変換
        3. 各トークンを ID にマッピング
           - もし語彙に無いトークンが出た場合は、文字単位に分解して再マッピングする
        """
        tokens = list(text)
        tokens = self._apply_merges(tokens)

        ids: list[int] = []
        for tok in tokens:
            if tok in self.token2id:
                ids.append(self.token2id[tok])
            else:
                # フォールバック:
                # 「: 」のような未知トークンは、1文字ずつに分解して
                # 既知の文字トークンとして扱う
                for ch in tok:
                    if ch in self.token2id:
                        ids.append(self.token2id[ch])
                    # 完全に未知の文字は捨てる（必要ならUNKトークンを追加してもよい）

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        トークンID列 → 文字列 に変換。
        各 ID をトークン文字列に戻し、すべて連結するだけ。
        """
        toks = [self.tokens[i] for i in ids]
        return "".join(toks)

    # ------------------------------------------------------------------
    # チェックポイント保存用
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        チェックポイント (.pt) に一緒に保存するためのシリアライズ用。
        - tokens: 語彙リスト（index = ID）
        - merges: マージ手順（[("に","ち"), ("こん","にち"), ...]）
        """
        return {
            "tokens": self.tokens,
            "merges": [(a, b) for (a, b) in self.merges],
        }


# ============================================
# ■ Dataset: 長いID列 → (入力, 正解) ペアに分割
# ============================================

"""
- 学習したいタスクは「次の1文字を当てる」。
- そのために、長いテキストをスライドさせながら

    入力:  [文字0, 文字1, ..., 文字(n-1)]
    正解:  [文字1, 文字2, ..., 文字n]

  のようなペアを大量に作る。

- PyTorch の Dataset クラスを継承して、自分専用のデータセットを作ると、
  DataLoader で自動的にミニバッチ化してくれる。
"""

class CharDataset(Dataset):
    def __init__(self, encoded_ids, block_size: int):
        super().__init__()
        self.block_size = block_size
        # リストで渡されたID列をPyTorchテンソルに変換
        self.data = torch.tensor(encoded_ids, dtype=torch.long)
        # 1サンプルで block_size+1 トークン使うので、最後に中途半端に余る部分はサンプルにできない
        self.num_samples = len(self.data) - block_size

    def __len__(self):
        # データセットの「長さ」＝何サンプルあるか
        return self.num_samples

    def __getitem__(self, idx):
        """
        idx番目のサンプルを取り出す。

        例:
          block_size = 4, data = [10,20,30,40,50,60] の場合

          idx=0:
            入力 x = [10,20,30,40]
            正解 y = [20,30,40,50]

          idx=1:
            入力 x = [20,30,40,50]
            正解 y = [30,40,50,60]
        """
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y


# ============================================
# ■ Transformer の構成要素：Multi-Head Self-Attention
# ============================================

"""
Self-Attention とは？
--------------------

- 入力された系列（例: 文章中の各単語/文字）について、
  「ある位置 i が、他のどの位置 j をどれだけ参照すべきか」
  を学習で決める仕組み。

- 具体的には、各位置に対し、
    - Query ベクトル Q
    - Key   ベクトル K
    - Value ベクトル V
  を用意し、

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V

  という計算を行う。

- GPT系では「因果マスク」を使い、**未来のトークンを見ない**ようにしている
  （時系列を壊さないため）。

Multi-Head とは？
-----------------

- 1つのHeadだけでなく、ベクトルを複数に分割し、
  複数の異なる視点で Self-Attention を計算するイメージ。
- 各Headは「異なる関係性」に注目できる（例：文法的依存、意味的類似など）。
"""

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size

        # Q, K, V を一気にまとめて作る線形層
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

        # ヘッドを結合した後に元の次元に戻す線形層
        self.out_proj = nn.Linear(d_model, d_model)

        # 因果マスク（未来のトークンを見ないようにする下三角マスク）
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask)

    def forward(self, x):
        """
        x: 形状 (B, T, C)
          B: バッチサイズ
          T: シーケンス長
          C: 特徴次元（d_model）
        """
        B, T, C = x.shape

        # 1. Q, K, V をまとめて計算
        qkv = self.qkv_proj(x)            # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)    # 3つに分割 → 各 (B, T, C)

        # 2. ヘッドごとに分割して (B, num_heads, T, head_dim) へ並び替え
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. スケーリング付きドット積 Attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, h, T, T)

        # 4. 因果マスク適用：未来（上三角）を -inf にして softmax で 0 になるようにする
        mask = self.mask[:T, :T]          # 実際の長さ T に合わせてマスクを切り出す
        att = att.masked_fill(mask == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)

        # 5. 注意重みで Value を混合
        y = att @ v                       # (B, h, T, head_dim)

        # 6. ヘッドを結合して元の形 (B, T, C) に戻し、最終線形層を通す
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return y


# ============================================
# ■ Transformer ブロック（Self-Attention + FFN）
# ============================================

"""
- Transformer ブロックは「Self-Attention」と「FFN（全結合ネットワーク）」を
  残差接続付きで組み合わせたもの。

構造（1ブロック）：
  x → LayerNorm → Self-Attention → 残差足し合わせ → x'
  x' → LayerNorm → FFN → 残差足し合わせ → 出力

- LayerNorm：各トークンのベクトルを正規化し、学習を安定させる
- 残差接続：入力を出力に足し合わせることで、勾配が深い層まで届きやすくなる
"""

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, block_size, ff_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, block_size)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x):
        # Self-Attention部分（事前にLayerNorm）
        x = x + self.attn(self.ln1(x))
        # FFN部分（再度LayerNorm）
        x = x + self.ff(self.ln2(x))
        return x


# ============================================
# ■ ミニGPT本体（埋め込み + Transformer層 + 出力層）
# ============================================

"""
MiniGPT の構造：

  1. 文字IDをベクトルに変換（token_emb）
  2. 位置情報を表すベクトルを加算（pos_emb）
  3. Transformer ブロックを複数層通す
  4. 最後に vocab_size 次元の線形層で「次の文字が各文字である確率の元（ロジット）」を出力する

- 入力の形: (B, T)
- 出力の形: (B, T, vocab_size)
"""

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, cfg: Config):
        super().__init__()
        self.block_size = cfg.block_size
        d_model = cfg.d_model

        # 文字埋め込み：ID → d_model 次元ベクトル
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # 位置埋め込み：位置ID（0〜block_size-1） → d_model 次元ベクトル
        self.pos_emb = nn.Embedding(cfg.block_size, d_model)

        # Transformer ブロックを num_layers 層積む
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=cfg.num_heads,
                    block_size=cfg.block_size,
                    ff_mult=cfg.ff_mult,
                )
                for _ in range(cfg.num_layers)
            ]
        )

        # 最後にかける LayerNorm
        self.ln_f = nn.LayerNorm(d_model)

        # 出力層：d_model → vocab_size
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        """
        idx: (B, T)  文字ID列
        """
        B, T = idx.shape
        assert T <= self.block_size

        # 文字埋め込み
        tok = self.token_emb(idx)                 # (B, T, d_model)

        # 位置埋め込み
        pos_ids = torch.arange(T, device=idx.device)
        pos = self.pos_emb(pos_ids)[None, :, :]   # (1, T, d_model)

        # 文字埋め込み＋位置埋め込み
        x = tok + pos

        # Transformerブロックを順に通す
        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.head(x)                    # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        すでにある文字列 idx から、max_new_tokens ぶんの文字を
        1文字ずつ追加していく生成用関数。
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ============================================
# ■ 学習ループ（train）
# ============================================

"""
目的：
  - 与えられたテキストから「次の1文字」を予測するようにモデルを学習させる。

大まかな流れ：
  1. トークナイザで文字をIDに変換
  2. Dataset / DataLoader で (入力, 正解) ペアをミニバッチ供給
  3. モデルに入力してロジットを得る
  4. CrossEntropyLoss で誤差を計算
  5. 誤差逆伝播で勾配を計算してパラメータを更新
  6. たまに文章生成してみて学習の進み具合を確認
  7. 各エポックの最後にチェックポイントを保存
"""

def train_model(text: str, cfg: Config, run_name: str):
    # 1. トークナイザ作成（テキストから BPE 語彙を学習）
    #    BPETokenizer は内部で「頻出ペアをマージする」処理を行う
    tokenizer = BPETokenizer(text)
    print(f"[info] vocab_size = {tokenizer.vocab_size}")

    # 2. テキスト全体を ID 列に変換
    encoded = tokenizer.encode(text)
    print(f"[info] num_tokens = {len(encoded)}")

    # 3. Dataset & DataLoader
    dataset = CharDataset(encoded, block_size=cfg.block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # 4. モデル・オプティマイザ・損失関数の用意
    model = MiniGPT(vocab_size=tokenizer.vocab_size, cfg=cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    global_step = 0

    # 5. 学習ループ（エポック × バッチ）
    for epoch in range(cfg.num_epochs):
        model.train()
        for x, y in dataloader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            logits = model(x)     # (B, T, vocab_size)

            # CrossEntropyLoss は (N, C) 形式を要求するので形を整える
            B, T, V = logits.shape
            loss = criterion(
                logits.view(B * T, V),
                y.view(B * T),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % 100 == 0:
                print(f"[epoch {epoch}] step {global_step} loss = {loss.item():.4f}")

            # 一定ステップごとに文章サンプルを生成して様子を見る
            if global_step % cfg.sample_every == 0:
                sample_text = generate_sample(model, tokenizer, cfg)
                print("=== sample ===")
                print(sample_text)
                print("==============")

            global_step += 1

        # 各エポックの最後にチェックポイントを保存
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = os.path.join("checkpoints", f"{run_name}_epoch{epoch}.pt")
        torch.save(
            {
                "model_state": model.state_dict(),
                # CharTokenizer では id2char を保存していたが、
                # BPE では「語彙 + マージ手順」の dict を保存する
                "tokenizer": tokenizer.to_dict(),
                "cfg": cfg.__dict__,
            },
            ckpt_path,
        )
        print(f"[info] saved checkpoint: {ckpt_path}")

    return model, tokenizer


@torch.no_grad()
def generate_sample(model: MiniGPT, tokenizer, cfg: Config, prompt: str = "これは"):
    """
    学習中や推論時に、簡単なサンプルテキストを生成するヘルパー関数。
    """
    model.eval()

    input_ids = tokenizer.encode(prompt)
    if len(input_ids) == 0:
        input_ids = [0]

    idx = torch.tensor([input_ids], dtype=torch.long, device=cfg.device)

    out = model.generate(
        idx,
        max_new_tokens=cfg.max_generate_tokens,
        temperature=cfg.temperature,
    )
    out_ids = out[0].tolist()
    return tokenizer.decode(out_ids)


# ============================================
# ■ 推論用：チェックポイントからモデル＆トークナイザを復元
# ============================================

def load_model_and_tokenizer(ckpt_path: str, device: str):
    """
    学習済みチェックポイント (.pt) から
    - Config
    - MiniGPT
    - BPETokenizer
    を復元するヘルパー関数。

    ※ BPE 版に変更したので、保存されているのは
      "vocab" ではなく "tokenizer"（語彙 + マージ手順）になっている。
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    tok_data = ckpt["tokenizer"]   # {"tokens": [...], "merges": [...]}
    cfg_dict = ckpt["cfg"]
    model_state = ckpt["model_state"]

    cfg = Config(**cfg_dict)
    cfg.device = device  # デバイスは今の環境に合わせて上書き

    # トークナイザを復元
    tokenizer = BPETokenizer(tok_data)

    # 語彙サイズは tokenizer 側の vocab_size を使う
    model = MiniGPT(vocab_size=tokenizer.vocab_size, cfg=cfg).to(device)
    model.load_state_dict(model_state)
    model.eval()

    return model, tokenizer, cfg


# ============================================
# ■ チャット風ループ（chat）
# ============================================

"""
- 完全な「対話モデル」ではなく、
  「会話ログを1つのテキストにして、続きを生成する」だけの簡易版。

内部イメージ：

  history = ""
  ユーザーが何か言うたびに、
      history += f"ユーザー: {発話}\nAI: "
      → ここから続きを生成
      → 生成されたテキストのうち、AIの返答らしき部分だけ抜き出して表示
      history += 返答 + "\n"

- モデルは「ユーザー:」「AI:」というトークンを学習していないので、
  会話としてはかなり適当な文章が返ってくるが、
  自作モデルと会話している“雰囲気”を楽しむことができる。
"""

def chat_loop(model: MiniGPT, tokenizer, cfg: Config, use_history: bool = True):
    """
    use_history:
        True  → これまでの会話ログ（history）を全部つないでプロンプトにする
        False → 毎ターン「ユーザー: 〜\nAI: 」だけをプロンプトにする（履歴なし）
    """
    print("=== mini_ja_gpt chat mode ===")
    print("終了したいとき: 空行 or 'exit' を入力して Enter。")
    print("---------------------------------")

    history = ""

    while True:
        user = input("あなた: ")
        if user.strip() == "" or user.strip().lower() == "exit":
            print("終了します。")
            break

        # プロンプト（モデルに渡すテキスト）を作る
        if use_history:
            # 履歴を全部つなげてコンテキストとして渡すモード
            history += f"ユーザー: {user}\nAI: "
            prompt_text = history
        else:
            # 毎ターン 1往復だけを見るモード（履歴オフ）
            prompt_text = f"ユーザー: {user}\nAI: "

        # プロンプトを ID 列に変換
        input_ids = tokenizer.encode(prompt_text)
        if len(input_ids) == 0:
            input_ids = [0]

        # block_size を超える分は最後だけ残す
        input_ids = input_ids[-cfg.block_size :]

        idx = torch.tensor([input_ids], dtype=torch.long, device=cfg.device)

        with torch.no_grad():
            out = model.generate(
                idx,
                max_new_tokens=cfg.max_generate_tokens,
                temperature=cfg.temperature,
            )

        out_ids = out[0].tolist()
        generated = tokenizer.decode(out_ids)

        # 生成結果から、「今の履歴部分」を削った分＝新規生成部分を取り出す
        base_text = tokenizer.decode(input_ids)
        new_part = generated[len(base_text) :]

        # 簡易的に「AIの返答っぽい部分」だけ抽出するために、
        # 改行や「ユーザー:」が出てきたところで切る
        cut_points = []
        if "\n" in new_part:
            cut_points.append(new_part.index("\n"))
        if "ユーザー:" in new_part:
            cut_points.append(new_part.index("ユーザー:"))
        reply = new_part
        if len(cut_points) > 0:
            reply = new_part[: min(cut_points)]

        reply = reply.strip()
        reply = reply[:200]  # 長すぎるときは適当にカット

        if reply == "":
            reply = "……（うまく生成できなかった）"

        print(f"AI: {reply}")

        if use_history:
            history += reply + "\n"


# ============================================
# ■ メイン関数：mode=train / mode=chat を切り替える
# ============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "chat"],
        help="train: 学習 / chat: チャット風推論",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="mode=train のときに使う学習テキストファイル (.txt)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="run",
        help="mode=train のとき、チェックポイント保存名のプレフィックス",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="mode=chat のときに読み込むチェックポイント .pt のパス",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="生成時の温度（1.0より大きいとランダム寄り、小さいと堅め）",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=200,
        help="1ターンあたり何文字ぶん生成するか（上限）",
    )
    parser.add_argument(
        "--no_history",
        action="store_true",
        help="指定すると、チャット時に履歴を使わず毎ターン単発で応答する",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")

    if args.mode == "train":
        # 学習モード
        if args.data_path is None:
            raise ValueError("--mode train には --data_path が必須です。")

        with open(args.data_path, "r", encoding="utf-8") as f:
            text = f.read()

        cfg = Config()
        cfg.device = device
        print(cfg)

        train_model(text, cfg, run_name=args.run_name)

    elif args.mode == "chat":
        # チャットモード
        if args.ckpt_path is None:
            raise ValueError("--mode chat には --ckpt_path が必須です。")

        model, tokenizer, cfg = load_model_and_tokenizer(args.ckpt_path, device)

        # 推論用パラメータ上書き
        cfg.temperature = args.temperature
        cfg.max_generate_tokens = args.max_tokens

        # --no_history が指定されていたら履歴を使わない
        use_history = not args.no_history
        chat_loop(model, tokenizer, cfg, use_history=use_history)


if __name__ == "__main__":
    # 乱数シード固定（再現性のため）
    torch.manual_seed(42)
    random.seed(42)
    main()