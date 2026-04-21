"""
good.py — メモリリークを修正した PyTorch 推論スクリプト

bad.py の 3 つの BUG をすべて修正したバージョン。
GPU メモリが推論回数によらず一定に保たれることを確認できる。

FIX:
  1. torch.no_grad() で計算グラフの構築を抑制
  2. テンソル参照を保持せず、結果の値だけを記録
  3. model.eval() で推論モードに切り替え
"""

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

# FIX 2: 結果の値だけを保持 (テンソル参照は持たない)
prediction_history: list[dict] = []


def create_dummy_image() -> Image.Image:
    """ダミー画像 (224x224 RGB) を生成する。"""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def load_model() -> torch.nn.Module:
    """モデルをロードする。"""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = model.to(DEVICE)

    # FIX 3: eval() で推論モードに切り替え
    #   → BatchNorm は running_mean/var を使用 (安定)
    #   → Dropout は無効化
    model.eval()

    return model


def predict(model: torch.nn.Module, image: Image.Image) -> dict:
    """1 枚の画像に対して推論を実行する。"""
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # FIX 1: torch.no_grad() で計算グラフの構築を抑制
    #   → 中間テンソルが保持されない
    #   → GPU メモリが推論ごとに増加しない
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        top5_prob, top5_idx = torch.topk(probs, 5, dim=1)

    result = {
        "top5_classes": top5_idx.squeeze(0).cpu().tolist(),
        "top5_probabilities": top5_prob.squeeze(0).cpu().tolist(),
    }

    # FIX 2: 値だけを保持。テンソル参照は保持しない。
    prediction_history.append(result)

    # テンソルを明示的に解放 (任意だが、メモリ断片化の予防になる)
    del input_tensor, output, probs, top5_prob, top5_idx

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
    print("RESULT (GOOD)")
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
