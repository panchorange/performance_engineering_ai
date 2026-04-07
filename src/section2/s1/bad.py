"""
bad.py — メモリリークする PyTorch 推論スクリプト

ResNet-50 で画像分類推論を繰り返す。
以下の問題により、推論回数に比例して GPU メモリが増加し続け、
最終的に OOM で停止する。

BUG:
  1. torch.no_grad() 未使用 → autograd 計算グラフが毎回蓄積
  2. 推論結果のテンソル参照を履歴リストに保持 → GC されない
  3. model.eval() 未呼び出し → BatchNorm/Dropout が訓練モード動作
"""

import io
import time

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
NUM_ITERATIONS = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# BUG 2: 推論結果を無制限に保持するリスト
prediction_history: list[dict] = []


def create_dummy_image() -> Image.Image:
    """ダミー画像 (224x224 RGB) を生成する。"""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def load_model() -> torch.nn.Module:
    """モデルをロードする。"""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = model.to(DEVICE)

    # BUG 3: model.eval() を呼ばない
    #   → BatchNorm がバッチ統計量を使用 (不安定)
    #   → Dropout が確率的にニューロンを落とす
    # 正しくは: model.eval()

    return model


def predict(model: torch.nn.Module, image: Image.Image) -> dict:
    """1 枚の画像に対して推論を実行する。"""
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # BUG 1: torch.no_grad() を使っていない
    #   → 推論なのに autograd が計算グラフを構築
    #   → 中間テンソル (活性化値、勾配用バッファ) が GPU メモリに保持される
    #   → 推論ごとにメモリが増加し続ける
    # 正しくは: with torch.no_grad():
    output = model(input_tensor)

    # CPU に転送して softmax (これ自体も非効率だが、メモリリークの主因ではない)
    probs = torch.softmax(output, dim=1)
    top5_prob, top5_idx = torch.topk(probs, 5, dim=1)

    result = {
        "top5_classes": top5_idx.squeeze(0).cpu().tolist(),
        "top5_probabilities": top5_prob.squeeze(0).detach().cpu().tolist(),
    }

    # BUG 2: テンソル参照を履歴に保持
    #   → output は計算グラフ全体への参照を持つ
    #   → GC されないため GPU メモリが解放されない
    # 正しくは: 結果の dict だけ保持するか、そもそも履歴を保持しない
    prediction_history.append({
        "result": result,
        "output_tensor": output,       # ← 計算グラフへの参照
        "input_tensor": input_tensor,   # ← GPU テンソルへの参照
    })

    return result


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Iterations: {NUM_ITERATIONS}")
    print()

    model = load_model()
    image = create_dummy_image()

    # 初期メモリ
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        mem_start = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"Initial GPU memory: {mem_start:.1f} MB")

    times: list[float] = []

    for i in range(NUM_ITERATIONS):
        t0 = time.perf_counter()
        predict(model, image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        if (i + 1) % 50 == 0:
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1024 / 1024
                print(
                    f"  [{i+1:4d}/{NUM_ITERATIONS}] "
                    f"latency={elapsed*1000:.1f}ms  "
                    f"GPU mem={mem:.1f}MB  "
                    f"history_size={len(prediction_history)}"
                )
            else:
                print(
                    f"  [{i+1:4d}/{NUM_ITERATIONS}] "
                    f"latency={elapsed*1000:.1f}ms  "
                    f"history_size={len(prediction_history)}"
                )

    # サマリ
    print()
    print("=" * 60)
    print("RESULT (BAD)")
    print("=" * 60)
    print(f"  Avg latency:  {np.mean(times)*1000:.1f} ms")
    print(f"  p99 latency:  {np.percentile(times, 99)*1000:.1f} ms")
    print(f"  History size: {len(prediction_history)}")

    if torch.cuda.is_available():
        mem_end = torch.cuda.memory_allocated() / 1024 / 1024
        mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  GPU mem start:  {mem_start:.1f} MB")
        print(f"  GPU mem end:    {mem_end:.1f} MB")
        print(f"  GPU mem peak:   {mem_peak:.1f} MB")
        print(f"  GPU mem leaked: {mem_end - mem_start:.1f} MB")


if __name__ == "__main__":
    main()
