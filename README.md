# Maia-2 Shogi

Maia-2 ([arxiv](https://arxiv.org/abs/2409.20553), [github](https://github.com/CSSLab/maia2))の将棋バージョン

学習データには将棋倶楽部２４のR800~R2800までの棋譜、計5億局面を使用しています。

## セットアップ

このプロジェクトは[uv](https://github.com/astral-sh/uv)を使用します。

```bash
# uvのインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係のインストール
uv sync
```

## モデルのダウンロード

学習済みONNXモデルは[Releases](https://github.com/nyoki-mtl/maia2-shogi/releases)からダウンロードしてください。

## クイックスタート

```bash
# 1. リポジトリをクローン
git clone https://github.com/YOUR_USERNAME/maia2-shogi.git
cd maia2-shogi

# 2. 依存関係をインストール
uv sync

# 3. サンプルを実行
uv run scripts/run_maia2_visualize_samples.py \
    model.onnx \
    data/example_sfen.txt \
    --output my_visualization.html
```

## 使い方

### 1. 単一局面の予測

```bash
uv run scripts/run_maia2_predict_sfen.py \
    model.onnx \
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1" \
    --rating-self 1500 \
    --top-k 5
```

### 2. 複数局面の可視化（レーティング比較）

```bash
uv run scripts/run_maia2_visualize_samples.py \
    models/model.onnx \
    data/example_sfen.txt \
    --output visualization.html \
    --top-k 5
```
