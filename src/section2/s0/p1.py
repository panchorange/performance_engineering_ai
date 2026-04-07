import torch
import torch.nn as nn

from torch.profiler import profile, ProfilerActivity, record_function

model = nn.Sequential(
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# ダミーデータ
input_data = torch.randn(64, 1024)

# プロファイリングの実行
with profile(activities=[
    ProfilerActivity.CPU,
    ProfilerActivity.CUDA,
], with_stack=True) as prof:
    with record_function("forward"):
        output = model(input_data)
    with record_function("loss_calc"):
        loss = output.sum()
    with record_function("backward"):
        loss.backward()


# 結果の表示
print(prof.key_averages().table(
    sort_by="cpu_time_total",
    row_limit=10
))

# Chrome Traceの出力
prof.export_chrome_trace("trace.json")
print("trace.json を出力しました。chrome://tracing で開いてください。")



