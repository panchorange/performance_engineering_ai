"""
Mixture of Experts (MoE) の簡単な実装
Router の動作を理解するための教育用コード

アーキテクチャ:
  入力 → Router → Top-K Expert 選択 → Expert 計算 → 重み付き合算 → 出力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Expert: 単純な FFN
# ─────────────────────────────────────────────
class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


# ─────────────────────────────────────────────
# 2. Router: どの Expert に routing するか決める
# ─────────────────────────────────────────────
class Router(nn.Module):
    """
    Router の役割:
      - 各トークンに対して「どの Expert を使うべきか」を決定する
      - Linear 層でスコアを計算 → Softmax で確率化 → Top-K を選択
      - 選ばれた Expert の確率がゲート値(重み)になる
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        # d_model → num_experts のスコアを出す Linear 層
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            top_k_weights:  (batch, seq_len, top_k)   各 Expert の重み (Softmax 後)
            top_k_indices:  (batch, seq_len, top_k)   選ばれた Expert のインデックス
            router_logits:  (batch, seq_len, num_experts)  生スコア (損失計算用)
        """
        # スコア計算: (batch, seq_len, num_experts)
        router_logits = self.gate(x)

        # Softmax で確率化
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-K を選択
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Top-K 内で再正規化（選ばれた Expert の確率の合計を 1 にする）
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        return top_k_weights, top_k_indices, router_logits


# ─────────────────────────────────────────────
# 3. MoE 層
# ─────────────────────────────────────────────
class MoELayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = Router(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
            aux_info: Router の内部情報（デバッグ・可視化用）
        """
        batch, seq_len, d_model = x.shape

        top_k_weights, top_k_indices, router_logits = self.router(x)

        # (batch * seq_len, d_model) に reshape して Expert を適用する
        x_flat = x.view(-1, d_model)  # (B*T, d_model)
        weights_flat = top_k_weights.view(-1, self.top_k)   # (B*T, top_k)
        indices_flat = top_k_indices.view(-1, self.top_k)   # (B*T, top_k)

        output_flat = torch.zeros_like(x_flat)  # (B*T, d_model)

        # Expert ごとにまとめて処理（token dispatch）
        for k in range(self.top_k):
            expert_idx_per_token = indices_flat[:, k]  # (B*T,)
            weight_per_token = weights_flat[:, k]       # (B*T,)

            for e in range(self.num_experts):
                # この Expert を使うトークンのマスク
                mask = (expert_idx_per_token == e)
                if not mask.any():
                    continue

                expert_input = x_flat[mask]                        # (n_tok, d_model)
                expert_output = self.experts[e](expert_input)      # (n_tok, d_model)
                # ゲート重みをスケールして加算
                output_flat[mask] += weight_per_token[mask].unsqueeze(-1) * expert_output

        output = output_flat.view(batch, seq_len, d_model)

        aux_info = {
            "router_logits": router_logits,          # 生スコア
            "top_k_indices": top_k_indices,          # 選択された Expert
            "top_k_weights": top_k_weights,          # ゲート重み
        }
        return output, aux_info


# ─────────────────────────────────────────────
# 4. Router の動作を可視化
# ─────────────────────────────────────────────
def visualize_routing(aux_info: dict, num_experts: int):
    """各 Expert が何トークンを担当したか（Expert Load）を表示"""
    indices = aux_info["top_k_indices"]  # (batch, seq_len, top_k)
    weights = aux_info["top_k_weights"]  # (batch, seq_len, top_k)

    # Expert ごとのトークン数をカウント
    flat_indices = indices.view(-1).tolist()
    from collections import Counter
    load = Counter(flat_indices)

    total_slots = indices.numel()
    print("\n=== Expert Load (Router の routing 結果) ===")
    for e in range(num_experts):
        count = load.get(e, 0)
        bar = "█" * count
        print(f"  Expert {e:2d}: {bar} ({count}/{total_slots} = {count/total_slots:.1%})")

    # 各トークンのゲート重み（最初のバッチ、最初の5トークン）
    print("\n=== Gate Weights (最初の5トークン) ===")
    for t in range(min(5, indices.shape[1])):
        idxs = indices[0, t].tolist()
        wgts = weights[0, t].tolist()
        chosen = ", ".join(f"Expert{i}({w:.3f})" for i, w in zip(idxs, wgts))
        print(f"  token[{t}] → {chosen}")


# ─────────────────────────────────────────────
# 5. 動作確認
# ─────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)

    # ハイパーパラメータ
    BATCH      = 2
    SEQ_LEN    = 8
    D_MODEL    = 16
    D_FF       = 32
    NUM_EXPERTS = 4
    TOP_K       = 2   # 各トークンが使う Expert 数

    moe = MoELayer(D_MODEL, D_FF, NUM_EXPERTS, TOP_K)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    print(f"Input shape : {x.shape}")
    output, aux_info = moe(x)
    print(f"Output shape: {output.shape}")

    visualize_routing(aux_info, NUM_EXPERTS)

    # Router の生スコア（logits）も確認
    logits = aux_info["router_logits"][0]  # (seq_len, num_experts)
    print(f"\n=== Router Logits (batch=0) ===")
    print(logits.detach().round(decimals=3))
