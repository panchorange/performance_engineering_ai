"""
bad.py — GPU推論 + Prometheusメトリクス（no_grad なしバージョン）

BUG: torch.no_grad() 未使用
  → autograd が計算グラフ + 中間活性化テンソルをVRAMに確保
  → バッチが大きいほどVRAM消費・レイテンシへの影響が顕著になる

起動方法:
  uv run src/section2/s2/bad.py

Prometheusエンドポイント: http://localhost:8000/metrics
"""

import time

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from prometheus_client import Gauge, start_http_server

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32  # バッチを大きくすることでno_gradの効果が顕著になる

LATENCY_MS    = Gauge("inference_latency_ms", "Batch latency in milliseconds")
THROUGHPUT    = Gauge("inference_throughput_rps", "Throughput in images/sec")
VRAM_PEAK_MB  = Gauge("inference_vram_peak_mb", "Peak VRAM per iteration in MB")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model() -> torch.nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    return model.to(DEVICE)


def predict(model: torch.nn.Module, batch: torch.Tensor) -> None:
    # BUG: torch.no_grad() を使っていない
    #   → autograd が計算グラフと中間活性化テンソルをVRAMに確保
    #   → batch_size=32 では数百MB単位でVRAMオーバーヘッドが発生
    # 正しくは: with torch.no_grad():
    model(batch)


def main() -> None:
    start_http_server(8000)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print("Prometheus metrics: http://localhost:8000/metrics")
    print("Ctrl+C で停止")

    model = load_model()

    dummy = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    single = preprocess(dummy).unsqueeze(0).to(DEVICE)
    batch = single.repeat(BATCH_SIZE, 1, 1, 1)

    # GPUウォームアップ（no_grad付きで行い、計測に影響させない）
    with torch.no_grad():
        for _ in range(5):
            model(batch)
    torch.cuda.synchronize()

    i = 0
    while True:
        t0 = time.perf_counter()
        predict(model, batch)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        rps = (BATCH_SIZE * 1000) / elapsed_ms
        vram_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()  # 次のイテレーション用にリセット

        LATENCY_MS.set(elapsed_ms)
        THROUGHPUT.set(rps)
        VRAM_PEAK_MB.set(vram_peak)

        if i % 50 == 0:
            print(f"  [{i:5d}] latency={elapsed_ms:.1f}ms  {rps:.0f}img/s  vram_peak={vram_peak:.0f}MB")

        i += 1


if __name__ == "__main__":
    main()
