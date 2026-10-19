# 原论文协议下的完整正式消融

TS/OS各五折×三个seed；Fit为作者TRAIN＋VAL，最终Test矩阵一次统一计分，历史Test已曝光。RF固定每fold seed42作为重复配对参照，不是三次独立RF拟合。

## TS

| 方法 | 五折三seed宏F1% | 有效完整比较 |
|---|---:|---|
| RF | 73.705002 | True |
| Base | 81.626778 | True |
| BiasPair | 81.715151 | True |
| ContextPair | 81.755051 | True |
| NoMotionPair | 81.661369 | True |
| MotionPair | 82.489275 | True |
| RecurrentOnly | 81.284734 | True |
| R7Fine.global7 | 82.972531 | True |
| R7Fine.pair7 | 82.452723 | True |
| R7Fine.pair3 | 82.444273 | True |
| CurrentOnly | 72.572632 | True |
| FullUniform | 81.878855 | True |
| FullBalanced | 78.265207 | True |
| NoUWB | 64.541675 | True |
| NoHD | 77.511470 | True |
| NoAkl | 80.912995 | True |
| RecurrentOnlyMotion | 82.278388 | True |
| CurrentOnlyMotion | 77.283663 | True |
| FullUniformMotion | 82.927266 | True |
| FullBalancedMotion | 79.585874 | True |
| NoUWBMotion | 64.593386 | True |
| NoHDMotion | 78.285728 | True |
| NoAklMotion | 81.866353 | True |
| R7Coarse.global7 | 81.186665 | True |
| R7Coarse.pair7 | 82.484893 | True |
| R7Coarse.pair3 | 82.477757 | True |
| LinearPair | 82.424746 | False |
| SpeedThresholdPair | 82.620647 | True |

| 完整主线去一项 | Macro差值pp | 条件95%区间pp |
|---|---:|---|
| Base | +0.862497 | [0.6765245921915097, 1.060021315718098] |
| RecurrentOnlyMotion | +0.210887 | [-0.030180367475707844, 0.48155220503461116] |
| CurrentOnlyMotion | +5.205612 | [4.270696603278245, 6.052698856151398] |
| FullUniformMotion | -0.437991 | [-1.1618000498449392, 0.1691314165135649] |
| FullBalancedMotion | +2.903401 | [2.3536252893798557, 3.3917097330777564] |
| NoUWBMotion | +17.895889 | [15.724054079763935, 19.85406449022714] |
| NoHDMotion | +4.203547 | [3.4739122881637434, 5.033042343554636] |
| NoAklMotion | +0.622923 | [0.24467909385299108, 0.9653274403530969] |
| LinearPair | +0.064530 | 未决 |
| SpeedThresholdPair | -0.131371 | [-0.5178625526553247, 0.23322517883006666] |

全部对比/逐类/逐fold/seed/cow、辅助pooled、错误流及交互保存于JSON，不隐去负值。
## OS

| 方法 | 五折三seed宏F1% | 有效完整比较 |
|---|---:|---|
| RF | 71.362433 | True |
| Base | 78.822512 | True |
| BiasPair | 78.865644 | True |
| ContextPair | 78.977350 | True |
| NoMotionPair | 79.000601 | True |
| MotionPair | 79.861014 | True |
| RecurrentOnly | 78.482421 | True |
| CurrentOnly | 70.766468 | True |
| FullUniform | 78.390877 | True |
| FullBalanced | 76.511639 | True |
| NoUWB | 60.668421 | True |
| NoHD | 77.477167 | True |
| NoAkl | 78.370350 | True |
| RecurrentOnlyMotion | 79.434252 | True |
| CurrentOnlyMotion | 75.674976 | True |
| FullUniformMotion | 79.370104 | True |
| FullBalancedMotion | 77.739603 | True |
| NoUWBMotion | 60.647890 | True |
| NoHDMotion | 78.263626 | True |
| NoAklMotion | 79.358759 | True |
| R7Fine.global7 | 79.961719 | True |
| R7Fine.pair7 | 79.838456 | True |
| R7Fine.pair3 | 79.833298 | True |
| R7Coarse.global7 | 79.373995 | True |
| R7Coarse.pair7 | 79.946028 | True |
| R7Coarse.pair3 | 79.935122 | True |
| LinearPair | 79.557662 | False |
| SpeedThresholdPair | 80.100152 | True |

| 完整主线去一项 | Macro差值pp | 条件95%区间pp |
|---|---:|---|
| Base | +1.038502 | [0.8426042666927541, 1.2319004119302057] |
| RecurrentOnlyMotion | +0.426763 | [-0.043228951393976256, 0.7656202203182169] |
| CurrentOnlyMotion | +4.186038 | [3.3016739873255636, 5.301061895795074] |
| FullUniformMotion | +0.490910 | [-0.11685504466573764, 1.1473857997010721] |
| FullBalancedMotion | +2.121412 | [1.511394083651762, 2.687666271149413] |
| NoUWBMotion | +19.213125 | [17.680037579259174, 21.006345291497517] |
| NoHDMotion | +1.597388 | [0.6784415058777621, 2.5294382540524523] |
| NoAklMotion | +0.502256 | [0.06066848298578345, 0.9199185980918108] |
| LinearPair | +0.303352 | 未决 |
| SpeedThresholdPair | -0.239138 | [-0.5977508856723083, 0.12322722891000058] |

全部对比/逐类/逐fold/seed/cow、辅助pooled、错误流及交互保存于JSON，不隐去负值。

固定模型/名单条件区间，不恢复盲性、不声称新牛或场地总体泛化；多项区间未校正，不称全家族显著。Mask改变有效容量；统计改善不等于独特创新。