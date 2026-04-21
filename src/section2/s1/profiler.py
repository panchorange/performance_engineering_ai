"""
profile.py — PyTorch Profiler で bad.py / good.py の差を可視化する

BAD (no_grad なし, テンソル保持) と GOOD (修正済み) の推論を
PyTorch Profiler で計測し、以下を比較出力する:
  - CPU / CUDA 時間の上位オペレータ
  - メモリ使用量の上位オペレータ
  - 推論を繰り返した際の GPU メモリ増加曲線

出力:
  - コンソールにサマリテーブル
  - ./profiler_output/ に Chrome Trace (chrome://tracing で閲覧)
"""

import io
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.profiler import profile, record_function, ProfilerActivity
from PIL import Image

# ---------------------------------------------------------------------------
# 共通設定
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("profiler_output")
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_WARMUP = 5
NUM_PROFILE = 30
NUM_MEMORY_ITERATIONS = 200

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def create_dummy_image() -> Image.Image:
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ============================================================================
# BAD: bad.py と同じ推論ロジック
# ============================================================================
_history_bad: list[dict] = []


def predict_bad(model: torch.nn.Module, image: Image.Image) -> dict:
    with record_function("BAD_predict"):
        with record_function("BAD_preprocess"):
            input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        # no_grad なし → autograd グラフ構築
        with record_function("BAD_forward [no_grad MISSING]"):
            output = model(input_tensor)

        with record_function("BAD_postprocess"):
            probs = torch.softmax(output, dim=1)
            top5_prob, top5_idx = torch.topk(probs, 5, dim=1)
            result = {
                "top5_classes": top5_idx.squeeze(0).cpu().tolist(),
                "top5_probabilities": top5_prob.squeeze(0).detach().cpu().tolist(),
            }

        # テンソル参照を保持 → メモリリーク
        with record_function("BAD_history_append [LEAK]"):
            _history_bad.append({
                "output_tensor": output,
                "input_tensor": input_tensor,
            })

    return result


# ============================================================================
# GOOD: good.py と同じ推論ロジック
# ============================================================================
def predict_good(model: torch.nn.Module, image: Image.Image) -> dict:
    with record_function("GOOD_predict"):
        with record_function("GOOD_preprocess"):
            input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        with record_function("GOOD_forward [no_grad]"):
            with torch.no_grad():
                output = model(input_tensor)

        with record_function("GOOD_postprocess"):
            with torch.no_grad():
                probs = torch.softmax(output, dim=1)
                top5_prob, top5_idx = torch.topk(probs, 5, dim=1)
            result = {
                "top5_classes": top5_idx.squeeze(0).cpu().tolist(),
                "top5_probabilities": top5_prob.squeeze(0).cpu().tolist(),
            }

        with record_function("GOOD_cleanup"):
            del input_tensor, output, probs, top5_prob, top5_idx

    return result


# ============================================================================
# プロファイル実行
# ============================================================================
def run_profile(label: str, predict_fn, model: torch.nn.Module, image: Image.Image) -> None:
    print(f"\n{'='*70}")
    print(f" Profiling: {label} ({NUM_WARMUP} warmup + {NUM_PROFILE} profiled)")
    print(f"{'='*70}")

    # ウォームアップ
    for _ in range(NUM_WARMUP):
        predict_fn(model, image)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(NUM_PROFILE):
            predict_fn(model, image)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # --- CPU 時間 ---
    print(f"\n  [{label}] CPU time (top 15)")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    # --- CUDA 時間 ---
    if torch.cuda.is_available():
        print(f"\n  [{label}] CUDA time (top 15)")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    # --- メモリ ---
    print(f"\n  [{label}] Memory usage (top 15)")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15))

    # Chrome Trace 出力
    trace_path = OUTPUT_DIR / f"{label}_trace.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"  Chrome trace: {trace_path}")


# ============================================================================
# メモリ増加曲線の計測
# ============================================================================
def measure_memory_growth(
    label: str,
    predict_fn,
    model: torch.nn.Module,
    image: Image.Image,
) -> list[float]:
    print(f"\n{'='*70}")
    print(f" Memory growth: {label} ({NUM_MEMORY_ITERATIONS} iterations)")
    print(f"{'='*70}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    mem_log: list[float] = []

    for i in range(NUM_MEMORY_ITERATIONS):
        predict_fn(model, image)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            import psutil
            mem_mb = psutil.Process().memory_info().rss / 1024 / 1024

        mem_log.append(mem_mb)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1:4d}/{NUM_MEMORY_ITERATIONS}] memory = {mem_mb:.1f} MB")

    return mem_log


# ============================================================================
# メイン
# ============================================================================
def main() -> None:
    print(f"Device: {DEVICE}")

    # --- モデル準備 ---
    print("Loading ResNet-50 (BAD: no eval)...")
    model_bad = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(DEVICE)
    # eval() なし (BUG 3 再現)

    print("Loading ResNet-50 (GOOD: eval)...")
    model_good = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(DEVICE)
    model_good.eval()

    image = create_dummy_image()

    # =============================================
    # STEP 1: PyTorch Profiler で比較
    # =============================================
    run_profile("BAD", predict_bad, model_bad, image)

    # BAD の履歴をクリアして GOOD に影響しないようにする
    _history_bad.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    run_profile("GOOD", predict_good, model_good, image)

    # =============================================
    # STEP 2: メモリ増加曲線の比較
    # =============================================
    _history_bad.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mem_bad = measure_memory_growth("BAD", predict_bad, model_bad, image)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mem_good = measure_memory_growth("GOOD", predict_good, model_good, image)

    # =============================================
    # STEP 3: サマリ
    # =============================================
    print(f"\n{'='*70}")
    print(" SUMMARY")
    print(f"{'='*70}")

    bad_delta = mem_bad[-1] - mem_bad[0]
    good_delta = mem_good[-1] - mem_good[0]

    print(f"  BAD  : {mem_bad[0]:.1f} MB → {mem_bad[-1]:.1f} MB  (delta: +{bad_delta:.1f} MB)")
    print(f"  GOOD : {mem_good[0]:.1f} MB → {mem_good[-1]:.1f} MB  (delta: +{good_delta:.1f} MB)")
    print()

    if bad_delta > good_delta * 5 + 1:
        print("  ** BAD パスで明確なメモリリークを検出 **")
        print("     原因: torch.no_grad() の欠如 + テンソル参照の保持")
    elif bad_delta > good_delta + 1:
        print("  ** BAD パスでメモリ増加傾向あり **")
    else:
        print("  (差が小さい場合: iteration 数を増やして再実行してください)")

    print()
    print("  確認方法:")
    print(f"    chrome://tracing → {OUTPUT_DIR}/BAD_trace.json と GOOD_trace.json を比較")
    print()


if __name__ == "__main__":
    main()
