"""
Maia-2 推論可視化スクリプト (ONNX版)

指定したSFEN局面に対して、複数のレーティング帯でONNX形式のMaia-2モデルを用いて推論し、
各局面の盤面・レーティング帯別の予測をHTMLとしてレポート化する。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from pathlib import Path

import cshogi
import numpy as np
import onnxruntime as ort
from cshogi import KI2
from cshogi.dlshogi import FEATURES1_NUM, FEATURES2_NUM, make_input_features, make_move_label
from tqdm.auto import tqdm

RATING_BASE = 800
RATING_STEP = 100
RATING_BINS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Maia-2 ONNX predictions on SFEN positions.")
    parser.add_argument("onnx_path", type=Path, help="Path to Maia-2 ONNX model.")
    parser.add_argument("sfen_path", type=Path, help="Path to SFEN file (one position per line).")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top policy moves to display for each sample (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="maia2_visualization.html",
        help="HTML output path (default: %(default)s).",
    )
    return parser.parse_args()


def bin_rating(rating: int) -> int:
    """レーティング値をビンインデックスに変換"""
    if rating < RATING_BASE or rating >= RATING_BASE + RATING_BINS * RATING_STEP:
        raise ValueError(
            f"レーティングは {RATING_BASE} <= rating < {RATING_BASE + RATING_BINS * RATING_STEP} の範囲で指定してください (got: {rating})."
        )
    return (rating - RATING_BASE) // RATING_STEP


def encode_board_dlshogi(board: cshogi.Board) -> np.ndarray:
    """cshogi.Boardから dlshogi特徴量を生成 (H, W, C)"""
    feature1 = np.zeros((FEATURES1_NUM, 9, 9), dtype=np.float32)
    feature2 = np.zeros((FEATURES2_NUM, 9, 9), dtype=np.float32)
    make_input_features(board, feature1, feature2)
    features = np.concatenate([feature1, feature2], axis=0)
    return np.transpose(features, (1, 2, 0))


def get_legal_moves_mask(board: cshogi.Board) -> np.ndarray:
    """合法手のマスクを生成 (2187,)"""
    mask = np.zeros(2187, dtype=np.float32)
    for move in board.legal_moves:
        label = make_move_label(move, board.turn)
        if 0 <= label < mask.size:
            mask[label] = 1.0
    return mask


def softmax_probabilities(logits: np.ndarray) -> np.ndarray:
    """ソフトマックスで確率に変換"""
    logits = logits.astype(np.float32, copy=False)
    max_logit = float(np.max(logits))
    shifted = logits.astype(np.float64, copy=False) - max_logit
    exp = np.exp(shifted, dtype=np.float64)
    sum_exp = float(np.sum(exp))
    if not np.isfinite(sum_exp) or sum_exp <= 0.0:
        return np.zeros_like(logits, dtype=np.float32)
    probs = exp / sum_exp
    return probs.astype(np.float32, copy=False)


def find_move_by_label(board: cshogi.Board, label: int) -> int | None:
    """ラベルから合法手を検索"""
    for move in board.legal_moves:
        generated = make_move_label(move, board.turn)
        if generated == label:
            return move
    return None


def rating_index_to_range(idx: int) -> tuple[int, int]:
    start = RATING_BASE + idx * RATING_STEP
    end = start + RATING_STEP - 1
    return start, end


def rating_index_to_label(idx: int) -> str:
    if idx < 0 or idx >= RATING_BINS:
        return f"{idx} (unknown)"
    r_min, r_max = rating_index_to_range(idx)
    return f"{idx} ({r_min}-{r_max})"


@dataclass
class RatingPrediction:
    """各レーティングでの予測結果"""

    rating: int  # レーティング値（例: 800, 1000, ...）
    rating_bin: int  # レーティングビン（0-19）
    policy_logits: np.ndarray  # (2187,)
    policy_probs: np.ndarray  # (2187,) softmax後
    value_logit: float
    value_prob: float  # sigmoid後


@dataclass
class PositionSample:
    """1つの局面とその複数レーティングでの予測"""

    index: int
    sfen: str
    board: cshogi.Board
    legal_moves: np.ndarray  # (2187,) bool mask
    rating_predictions: list[RatingPrediction]


def move_label_to_text(board: cshogi.Board, label: int) -> tuple[str, bool]:
    """ラベルから指し手文字列を取得"""
    for move in board.legal_moves:
        generated = make_move_label(move, board.turn)
        if generated == label:
            return KI2.move_to_ki2(move, board), True  # type: ignore[arg-type]
    return "N/A", False


def load_sfen_file(sfen_path: Path) -> list[str]:
    """SFENファイルから局面を読み込み（バリデーション付き）"""
    sfens_raw = []
    with sfen_path.open("r", encoding="utf-8") as f:
        sfens_raw = [line.strip() for line in f if line.strip()]

    sfens_valid = []
    for sfen in sfens_raw:
        sfens_valid.append(sfen)

    return sfens_valid


def predict_position_multi_ratings(
    session: ort.InferenceSession,
    board: cshogi.Board,
    legal_mask: np.ndarray,
    ratings: list[int],
    has_legal_moves_input: bool,
) -> list[RatingPrediction]:
    """1つの局面に対して複数のレーティングで推論"""
    features = encode_board_dlshogi(board).astype(np.float32)
    predictions: list[RatingPrediction] = []

    for rating in ratings:
        rating_bin = bin_rating(rating)
        inputs = {
            "board": features[np.newaxis, ...],
            "rating_self": np.array([rating_bin], dtype=np.int32),
            "rating_oppo": np.array([rating_bin], dtype=np.int32),
        }
        if has_legal_moves_input:
            inputs["legal_moves"] = legal_mask[np.newaxis, ...].astype(np.float32)

        outputs = session.run(None, inputs)
        if len(outputs) < 2:
            raise RuntimeError("ONNXモデルの出力が想定と異なります。policyとvalueが含まれる必要があります。")

        policy_logits = np.asarray(outputs[0])[0].astype(np.float32)
        value_logit = float(np.asarray(outputs[1]).reshape(-1)[0])

        # 合法手マスキング
        masked_logits = np.array(policy_logits, copy=True)
        masked_logits[legal_mask < 0.5] = -1e4
        policy_probs = softmax_probabilities(masked_logits)

        value_prob = float(1.0 / (1.0 + np.exp(-value_logit)))

        predictions.append(
            RatingPrediction(
                rating=rating,
                rating_bin=rating_bin,
                policy_logits=policy_logits,
                policy_probs=policy_probs,
                value_logit=value_logit,
                value_prob=value_prob,
            )
        )

    return predictions


def collect_predictions(
    session: ort.InferenceSession, sfens: list[str], ratings: list[int], has_legal_moves_input: bool
) -> list[PositionSample]:
    """全局面×全レーティングで予測を収集"""
    samples: list[PositionSample] = []

    for idx, sfen in enumerate(tqdm(sfens, desc="Predicting positions"), 1):
        board = cshogi.Board()
        try:
            board.set_sfen(sfen)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            print(f"\nWarning: SFEN解析失敗 (index {idx}): {sfen[:80]} - {exc}")
            continue

        try:
            legal_mask = get_legal_moves_mask(board)
            predictions = predict_position_multi_ratings(session, board, legal_mask, ratings, has_legal_moves_input)
        except Exception as exc:  # noqa: BLE001
            print(f"\nWarning: 予測失敗 (index {idx}): {sfen[:80]} - {exc}")
            continue

        samples.append(
            PositionSample(
                index=idx,
                sfen=sfen,
                board=board,
                legal_moves=legal_mask,
                rating_predictions=predictions,
            )
        )

    return samples


def generate_html(
    samples: list[PositionSample],
    onnx_path: Path,
    sfen_path: Path,
    top_k: int,
    output_path: Path,
) -> None:
    """HTML可視化レポートを生成（コンパクト版）"""
    html_parts: list[str] = []
    html_parts.append("<!DOCTYPE html>")
    html_parts.append('<html lang="ja">')
    html_parts.append("<head>")
    html_parts.append('<meta charset="utf-8" />')
    html_parts.append("<title>Maia-2 Multi-Rating Visualization (ONNX)</title>")
    html_parts.append(
        """
        <style>
        body { font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; margin: 20px; color: #333; font-size: 13px; }
        h1, h2, h3 { color: #1f2d3d; }
        h3 { font-size: 1.2em; margin-bottom: 0.5em; }
        table { border-collapse: collapse; margin: 1rem 0; font-size: 12px; }
        th, td { border: 1px solid #ddd; padding: 4px 8px; text-align: left; white-space: nowrap; }
        th { background-color: #f5f7fa; font-weight: bold; }
        .meta { margin-bottom: 1.5rem; font-size: 12px; }
        .position { border: 1px solid #ccc; padding: 12px; margin-bottom: 20px; border-radius: 4px; background: #fafafa; }
        .position h3 { margin-top: 0; }
        .board { margin: 0.5rem 0; }
        .compact-table { width: auto; }
        .compact-table th { text-align: center; }
        .compact-table td { text-align: left; font-size: 11px; }
        .compact-table td:first-child { font-weight: bold; text-align: center; background-color: #f9f9f9; }
        .small { font-size: 0.85em; color: #555; }
        </style>
        """
    )
    html_parts.append("</head>")
    html_parts.append("<body>")
    html_parts.append("<h1>Maia-2 Multi-Rating Visualization (ONNX)</h1>")
    html_parts.append('<div class="meta">')
    html_parts.append(f"<strong>Model:</strong> {escape(onnx_path.name)} | ")
    html_parts.append(f"<strong>SFEN:</strong> {escape(sfen_path.name)} | ")
    html_parts.append(f"<strong>Positions:</strong> {len(samples)}")
    html_parts.append("</div>")

    if not samples:
        html_parts.append("<p>No positions were loaded.</p>")
    else:
        for sample in samples:
            html_parts.append('<div class="position">')
            html_parts.append(f"<h3>Position {sample.index}</h3>")
            html_parts.append(f'<p class="small">{escape(sample.sfen)}</p>')

            try:
                svg = sample.board._repr_svg_()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                svg = f"<p class='small'>Failed to render board: {escape(str(exc))}</p>"

            html_parts.append(f'<div class="board">{svg}</div>')

            legal_indices = np.nonzero(sample.legal_moves)[0]
            legal_count = int(legal_indices.size)

            if legal_count == 0:
                html_parts.append("<p class='small'>No legal moves detected.</p>")
            else:
                # コンパクトなテーブル形式：行=レーティング、列=Value + Rank1-5
                html_parts.append('<table class="compact-table">')

                # ヘッダー行
                html_parts.append("<thead><tr>")
                html_parts.append("<th>R</th><th>Value</th>")
                for rank in range(1, min(top_k, legal_count) + 1):
                    html_parts.append(f"<th>Rank{rank}</th>")
                html_parts.append("</tr></thead>")

                html_parts.append("<tbody>")

                # 各レーティングの行
                for pred in sample.rating_predictions:
                    html_parts.append("<tr>")

                    # レーティング列
                    html_parts.append(f"<td>{pred.rating}</td>")

                    # Value列
                    html_parts.append(f"<td>{pred.value_prob:.3f}</td>")

                    # Top-K指し手
                    top_n = min(top_k, legal_count)
                    sorted_indices = legal_indices[np.argsort(pred.policy_probs[legal_indices])[::-1]]
                    top_indices = sorted_indices[:top_n]

                    for move_idx in top_indices:
                        move_idx = int(move_idx)
                        move = find_move_by_label(sample.board, move_idx)
                        if move is not None:
                            move_text = KI2.move_to_ki2(move, sample.board)  # type: ignore[arg-type]
                        else:
                            move_text = "N/A"
                        prob = float(pred.policy_probs[move_idx])
                        html_parts.append(f"<td>{escape(str(move_text))} ({prob:.3f})</td>")

                    html_parts.append("</tr>")

                html_parts.append("</tbody></table>")

            html_parts.append("</div>")

    html_parts.append("</body></html>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"Visualization saved to {output_path}")


def main() -> None:
    args = parse_args()
    onnx_path = args.onnx_path.expanduser().resolve()
    sfen_path = args.sfen_path.expanduser().resolve()

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNXファイルが見つかりません: {onnx_path}")
    if not sfen_path.exists():
        raise FileNotFoundError(f"SFENファイルが見つかりません: {sfen_path}")

    print(f"Loading ONNX model: {onnx_path}")
    session = ort.InferenceSession(onnx_path.as_posix(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    onnx_input_names = {input_.name for input_ in session.get_inputs()}
    has_legal_moves_input = "legal_moves" in onnx_input_names

    print(f"Loading SFEN file: {sfen_path}")
    sfens = load_sfen_file(sfen_path)
    print(f"Loaded {len(sfens)} positions")

    # レーティング: 800, 1000, 1200, ..., 2800
    ratings = list(range(RATING_BASE, RATING_BASE + RATING_BINS * RATING_STEP, RATING_STEP))
    print(f"Ratings to predict: {ratings}")

    samples = collect_predictions(session, sfens, ratings, has_legal_moves_input)

    output_path = Path(args.output)
    generate_html(
        samples=samples,
        onnx_path=onnx_path,
        sfen_path=sfen_path,
        top_k=args.top_k,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
