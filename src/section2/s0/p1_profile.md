# 出力結果
anch@pancho:~/study/performance_engineering_ai$ uv run src/section2/s0/p1.py 
/home/panch/study/performance_engineering_ai/.venv/lib/python3.12/site-packages/torch/profiler/profiler.py:224: UserWarning: Warning: Profiler clears events at the end of each cycle.Only events from the current cycle will be reported.To keep events across cycles, set acc_events=True.
  _warn_once(
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                           aten::linear         2.81%       1.092ms        47.40%      18.435ms       6.145ms             3
                                            aten::addmm        38.96%      15.152ms        41.17%      16.014ms       5.338ms             3
    autograd::engine::evaluate_function: AddmmBackward0         0.14%      52.655us        22.84%       8.881ms       2.960ms             3
                                         AddmmBackward0         7.19%       2.795ms        22.39%       8.708ms       2.903ms             3
                                           aten::linear         2.81%       1.092ms        47.40%      18.435ms       6.145ms             3
                                            aten::addmm        38.96%      15.152ms        41.17%      16.014ms       5.338ms             3
                                           aten::linear         2.81%       1.092ms        47.40%      18.435ms       6.145ms             3
                                            aten::addmm        38.96%      15.152ms        41.17%      16.014ms       5.338ms             3
    autograd::engine::evaluate_function: AddmmBackward0         0.14%      52.655us        22.84%       8.881ms       2.960ms             3
                                         AddmmBackward0         7.19%       2.795ms        22.39%       8.708ms       2.903ms             3
                                              aten::sum        13.63%       5.302ms        15.18%       5.905ms       1.476ms             4
                                               aten::mm        14.93%       5.808ms        15.03%       5.847ms       1.169ms             5
                                             aten::relu         2.31%     897.127us         5.52%       2.147ms       1.073ms             2
      autograd::engine::evaluate_function: SumBackward0         0.04%      16.783us         4.32%       1.681ms       1.681ms             1
                                           SumBackward0         1.92%     748.227us         4.28%       1.664ms       1.664ms             1
     autograd::engine::evaluate_function: ReluBackward0         0.03%      12.045us         3.82%       1.484ms     741.899us             2
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 38.892ms