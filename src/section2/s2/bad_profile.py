"""
bad_profile.py — bad.py の推論を torch.profiler で計測する

BUG: torch.no_grad() 未使用（bad.py と同じ）

起動方法:
  uv run src/section2/s2/bad_profile.py
"""

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    model = model.to(DEVICE)

    dummy = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    batch = preprocess(dummy).unsqueeze(0).repeat(BATCH_SIZE, 1, 1, 1).to(DEVICE)

    # ウォームアップ
    with torch.no_grad():
        for _ in range(5):
            model(batch)
    torch.cuda.synchronize()

    # --- プロファイリング（no_grad なし = bad） ---
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
    ) as prof:
        model(batch)  # ← no_grad なし
        torch.cuda.synchronize()

    print()
    print("=" * 80)
    print("CUDA時間 TOP10")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print()
    print("=" * 80)
    print("CUDAメモリ使用量 TOP10")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))


if __name__ == "__main__":
    main()
