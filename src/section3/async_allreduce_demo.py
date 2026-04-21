"""非同期 All-Reduce デモ

- CPU + gloo バックエンドで 4 プロセス起動（GPU 不要）
- 「擬似レイヤーの計算 → 勾配の All-Reduce」を N 回繰り返す
- 同期版と非同期版の総時間を比較する

必要なパッケージ:
    uv add torch

実行:
    uv run python src/section3/async_allreduce_demo.py
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# 必要なら環境に合わせて調整
WORLD_SIZE = 4
N_LAYERS = 10
TENSOR_SIZE = 4096  # 4096*4096*4B ≒ 64MB / テンソル
COMPUTE_ITERS = 3   # レイヤーあたりの計算負荷


def heavy_compute(x: torch.Tensor, iters: int = COMPUTE_ITERS) -> torch.Tensor:
    """計算カーネルの代役（行列積を数回）。時間を稼ぐ。"""
    for _ in range(iters):
        x = torch.matmul(x, x.T) / float(TENSOR_SIZE)
    return x


def worker(rank: int, world_size: int, mode: str, result_queue: "mp.Queue[float]") -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    # 各レイヤー分の「活性値」と「勾配」を用意
    acts = [torch.randn(TENSOR_SIZE, TENSOR_SIZE) for _ in range(N_LAYERS)]
    grads = [torch.randn(TENSOR_SIZE, TENSOR_SIZE) for _ in range(N_LAYERS)]

    # ウォームアップ（初回呼び出しコストを除外）
    for g in grads[:2]:
        dist.all_reduce(g)

    dist.barrier()
    t0 = time.time()

    if mode == "sync":
        # ❌ 同期: 通信が終わるまで次の計算に進めない
        for i in range(N_LAYERS):
            acts[i] = heavy_compute(acts[i])
            dist.all_reduce(grads[i])  # ここでブロック

    elif mode == "async":
        # ✅ 非同期: 通信を投げっぱなしにして次の計算を進める
        handles = []
        for i in range(N_LAYERS):
            acts[i] = heavy_compute(acts[i])
            h = dist.all_reduce(grads[i], async_op=True)  # すぐ帰ってくる
            handles.append(h)
        # 最後にまとめて同期
        for h in handles:
            h.wait()

    dist.barrier()
    elapsed = time.time() - t0

    if rank == 0:
        result_queue.put(elapsed)
        print(f"[{mode:5s}] 経過時間: {elapsed:.3f} 秒")

    dist.destroy_process_group()


def main() -> None:
    print(
        f"\n--- world_size={WORLD_SIZE}, backend=gloo(CPU), "
        f"layers={N_LAYERS}, tensor={TENSOR_SIZE}x{TENSOR_SIZE} ---\n"
    )

    times: dict[str, float] = {}
    for mode in ("sync", "async"):
        ctx = mp.get_context("spawn")
        q: "mp.Queue[float]" = ctx.Queue()
        mp.spawn(worker, args=(WORLD_SIZE, mode, q), nprocs=WORLD_SIZE, join=True)
        times[mode] = q.get()

    speedup = times["sync"] / times["async"]
    saved = (1 - times["async"] / times["sync"]) * 100
    print(f"\n🎯 スピードアップ: {speedup:.2f}x （{saved:.1f}% 削減）\n")


if __name__ == "__main__":
    main()
