"""
Maia-2 推論スクリプト (SFEN入力, ONNX推論用)

指定したSFEN局面に対してONNX形式のMaia-2モデルを用いて推論し、
policyのtop-k指し手とvalue予測をコンソールに出力する。
"""

import argparse
from pathlib import Path

import cshogi
import numpy as np
import onnxruntime as ort
from cshogi import KI2
from cshogi.dlshogi import FEATURES1_NUM, FEATURES2_NUM, make_input_features, make_move_label

# モデル設定（変更不可）
NUM_RATINGS = 20
MOVE_LABELS = 2187
RATE_MIN = 800
RATE_MAX = 2800
BIN_WIDTH = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Maia-2 ONNX inference on a single SFEN position.")
    parser.add_argument("onnx_path", type=Path, help="Path to Maia-2 ONNX model.")
    parser.add_argument("sfen", type=str, help="SFEN string representing the board position.")
    parser.add_argument(
        "--rating-self",
        type=int,
        default=1500,
        help="Player rating used for self (default: %(default)s).",
    )
    parser.add_argument(
        "--rating-oppo",
        type=int,
        default=None,
        help="Player rating used for opponent. Defaults to the same as --rating-self.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top policy moves to display (default: %(default)s).",
    )
    return parser.parse_args()


def bin_rating(rating: int) -> int:
    if rating < RATE_MIN or rating >= RATE_MAX:
        raise ValueError(f"レーティングは {RATE_MIN} <= rating < {RATE_MAX} の範囲で指定してください (got: {rating}).")
    return (rating - RATE_MIN) // BIN_WIDTH


def encode_board_dlshogi(board: cshogi.Board) -> np.ndarray:
    feature1 = np.zeros((FEATURES1_NUM, 9, 9), dtype=np.float32)
    feature2 = np.zeros((FEATURES2_NUM, 9, 9), dtype=np.float32)
    make_input_features(board, feature1, feature2)
    features = np.concatenate([feature1, feature2], axis=0)
    return np.transpose(features, (1, 2, 0))


def get_legal_moves_mask(board: cshogi.Board) -> np.ndarray:
    mask = np.zeros(MOVE_LABELS, dtype=np.float32)
    for move in board.legal_moves:
        label = make_move_label(move, board.turn)
        if 0 <= label < mask.size:
            mask[label] = 1.0
    return mask


def softmax_probabilities(logits: np.ndarray) -> np.ndarray:
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
    for move in board.legal_moves:
        generated = make_move_label(move, board.turn)
        if generated == label:
            return move
    return None


def main() -> None:
    args = parse_args()
    onnx_path = args.onnx_path.expanduser().resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNXファイルが見つかりません: {onnx_path}")

    rating_self = bin_rating(args.rating_self)
    rating_oppo_input = args.rating_oppo if args.rating_oppo is not None else args.rating_self
    rating_oppo = bin_rating(rating_oppo_input)

    board = cshogi.Board()
    try:
        board.set_sfen(args.sfen)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"SFENを解析できません: {args.sfen}") from exc

    features = encode_board_dlshogi(board).astype(np.float32)

    legal_mask = get_legal_moves_mask(board)

    session = ort.InferenceSession(onnx_path.as_posix(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    onnx_input_names = {input_.name for input_ in session.get_inputs()}
    inputs = {
        "board": features[np.newaxis, ...],
        "rating_self": np.array([rating_self], dtype=np.int32),
        "rating_oppo": np.array([rating_oppo], dtype=np.int32),
    }
    if "legal_moves" in onnx_input_names:
        inputs["legal_moves"] = legal_mask[np.newaxis, ...].astype(np.float32)

    outputs = session.run(None, inputs)
    if len(outputs) < 2:
        raise RuntimeError("ONNXモデルの出力が想定と異なります。policyとvalueが含まれる必要があります。")

    policy_logits = np.asarray(outputs[0])[0].astype(np.float32)
    value_logit = float(np.asarray(outputs[1]).reshape(-1)[0])

    masked_logits = np.array(policy_logits, copy=True)
    # 非合法手をマスキングしてからsoftmaxする
    masked_logits[legal_mask < 0.5] = -1e4
    policy_probs = softmax_probabilities(masked_logits)

    # # 2187ラベルにおいて、その手が現局面において合法手かどうかのconfidenceを出力する補助ヘッド
    # aux_logits = np.asarray(outputs[2])[0].astype(np.float32) if len(outputs) >= 3 else None
    # aux_probs = 1.0 / (1.0 + np.exp(-aux_logits)) if aux_logits is not None else None

    top_k = max(1, args.top_k)
    top_indices = np.argsort(policy_probs)[::-1][:top_k]

    print("=== Maia-2 ONNX Inference ===")
    print(f"ONNX model   : {onnx_path}")
    print(f"SFEN         : {args.sfen}")
    print(f"Rating (self): {args.rating_self} -> bin {rating_self}")
    print(f"Rating (oppo): {rating_oppo_input} -> bin {rating_oppo}")
    print(f"Legal moves  : {int(legal_mask.sum())}")
    print(f"Legal mask input provided to ONNX: {'legal_moves' in onnx_input_names}")
    print()

    print(f"Top-{len(top_indices)} policy moves:")
    for rank, label in enumerate(top_indices, 1):
        label = int(label)
        prob = float(policy_probs[label])
        move = find_move_by_label(board, label)
        if move is not None:
            move_text = KI2.move_to_ki2(move, board)  # type: ignore[arg-type]
        else:
            # 非合法手がきたらN/A。ただし非合法手はマスキングしてあるから、ここに来ることはないはず...
            move_text = "N/A"

        line = f" #{rank}: {move_text} ({prob:.6f})"
        print(line)

    value_prob = float(1.0 / (1.0 + np.exp(-value_logit)))
    print()
    print(f"Value : {value_prob:.6f}")


if __name__ == "__main__":
    main()
