# 十一模型完整比较

每模型先平均自身实际seed再五折等权；原RF单seed，其余三seed。

| 模型 | TS宏F1% | OS宏F1% |
| --- | ---: | ---: |
| Original-Paper-RF | 73.705002 | 71.362433 |
| E-GRU-SQRT-VAL | 81.301660 | 78.158506 |
| CurrentRepeat-GRU-VAL | 75.799211 | 74.769941 |
| Stats-RF-Window | 78.583326 | 77.775522 |
| Stats-RF-CurrentOnly | 75.803669 | 73.802211 |
| MiniRocket-Window | 70.215910 | 69.549616 |
| LSTM | 80.766670 | 78.573933 |
| Transformer | 81.170497 | 79.420856 |
| Transformer-LSTM | 80.815226 | 78.844461 |
| XGBoost-Window | 80.828917 | 77.185818 |
| TCN-Window-VAL | 81.157095 | 77.996871 |

本轮VAL-refit配方与原20轮主线分开；已有Test曝光，条件区间不包含重训/选型/新环境，未校正多重比较。
本总览用seed20261014重新统一计算区间；旧报告各自的seed及区间保持原件，不冒称两套bootstrap输出逐位相同。
既有审计证据引用与新结果审计范围分列，主线不自动替换。
