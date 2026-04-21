# S1: PyTorch 推論のメモリリーク

## テーマ

PyTorch で推論を繰り返すと GPU メモリが単調増加し、最終的に OOM で停止する問題の分析と修正。

## ファイル構成

| ファイル | 内容 |
|---------|------|
| `bad.py` | メモリリークする推論スクリプト |
| `good.py` | 修正済みの推論スクリプト |
| `profiler.py` | PyTorch Profiler で BAD / GOOD を比較分析 |
| `description.md` | 本ファイル |

## 実行方法

```bash
# 1. BAD 版を実行 → GPU メモリが増え続けることを確認
uv run python src/section2/s1/bad.py

# 2. GOOD 版を実行 → GPU メモリが安定していることを確認
uv run python src/section2/s1/good.py

# 3. Profiler で比較分析
uv run python src/section2/s1/profile.py
# → chrome://tracing で profiler_output/*.json を読み込み
```

---

## 埋め込まれた BUG (3つ)

### BUG 1: `torch.no_grad()` 未使用

```python
# BAD: autograd が計算グラフを構築し、中間テンソルが毎回蓄積
output = model(input_tensor)

# GOOD: 計算グラフを構築しない → メモリ増加なし
with torch.no_grad():
    output = model(input_tensor)
```

**影響**: ResNet-50 の 1 回の forward で数十 MB の計算グラフが生成される。
300 回推論すると数 GB のメモリが消費され、最終的に OOM。

**Profiler での確認方法**:
- `BAD_forward [no_grad MISSING]` の `self_cpu_memory_usage` が大きい
- メモリ増加曲線で BAD が右肩上がり、GOOD は横ばい

### BUG 2: テンソル参照の保持

```python
# BAD: 計算グラフへの参照を持つテンソルをリストに保持 → GC されない
prediction_history.append({
    "output_tensor": output,       # 計算グラフ全体が GC 不可に
    "input_tensor": input_tensor,
})

# GOOD: 値だけ保持。テンソル参照なし。
prediction_history.append(result)  # dict[str, list[float]]
del input_tensor, output
```

**影響**: BUG 1 と複合で、計算グラフが永遠に解放されなくなる。
BUG 1 だけなら GC で部分的に回収されうるが、参照を保持することで完全にリーク。

### BUG 3: `model.eval()` 未呼び出し

```python
# BAD: 訓練モードのまま
model = model.to(DEVICE)

# GOOD: 推論モードに切り替え
model = model.to(DEVICE)
model.eval()
```

**影響**:
- **BatchNorm**: バッチ統計量を使用するため、バッチサイズ 1 で結果が不安定
- **Dropout**: ニューロンをランダムに無効化 → 同じ入力でも異なる出力
- 直接メモリリークは起こさないが、結果の不安定さがリトライや障害の原因になる

---

## 実験で観測すべきポイント

### 1. GPU メモリの増加パターン

`bad.py` と `good.py` を実行し、50 イテレーションごとのメモリ出力を比較する。

**期待される結果**:
- BAD: `GPU mem` が 50 回ごとに数十〜数百 MB ずつ増加
- GOOD: `GPU mem` がほぼ一定 (モデルパラメータ分のみ)

### 2. Profiler のオペレータ比較

`profile.py` を実行し、以下に注目する:

| 比較項目 | BAD | GOOD |
|---------|-----|------|
| forward の CPU 時間 | autograd 分だけ遅い | 高速 |
| `self_cpu_memory_usage` | 大きい (グラフ分) | 小さい |
| CUDA カーネル | backward 用カーネルも起動 | forward のみ |

### 3. Chrome Trace での視覚比較

`chrome://tracing` で BAD/GOOD の trace.json を開き:
- BAD: 各 iteration で `aten::*` の backward 関連オペレータが並ぶ
- GOOD: forward のオペレータのみで完結

---

## なぜリリース直後は問題にならなかったか

1. **メモリ蓄積は緩やか**: 1 推論あたり数十 MB。GPU メモリ (5070 Ti: 16 GB) に対して最初の数百回は余裕がある
2. **トラフィックが少ないうちは発症しない**: 1 日数百リクエストなら数 GB 程度で、サービス再起動で回収される
3. **eval() の問題は気づきにくい**: 結果は「だいたい合っている」ので、精度のわずかな低下は見過ごされがちい

運用が本格化し、リクエスト数が増加すると問題が顕在化する。

---

## 修正の優先順位

| 優先度 | BUG | 理由 |
|:-----:|:---:|------|
| **P0** | 1 + 2 | OOM でサービス停止に直結。セットで修正が必要 |
| **P1** | 3 | 推論精度と安定性に影響。本番で気づきにくいが重要 |
