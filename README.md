# 奶牛行为识别论文审稿资料

对应中文稿《融合多源短时历史信息与当前状态的奶牛行为识别》，v10。此仓库用于审稿、核查方法实现、复算保存结果与查看图表来源。

固定版本标签：`paper-v10-20261019`。本仓库不是新增数据采集，原始数据来自MmCows。

## 论文与补充材料

- 论文主稿：[Word](manuscript/manuscript_v10_zh.docx) / [PDF预览](manuscript/manuscript_v10_zh.pdf)
- 补充材料：[Word](manuscript/supplement_v10_zh.docx) / [PDF预览](manuscript/supplement_v10_zh.pdf)，集中表S1—S7。
- [图1—图5的SVG](figures/manuscript/)
- [材料与论文表格对应关系](docs/ARTIFACT_MAP.md)

## 两级离线核查

Python 3.9或更新版本，先核对文件完整性：

```bash
python verify_manifest.py
python verify_saved_results.py
```

以上只使用标准库，从保存的混淆矩阵复算主表、固定20轮结果和目标/划分规模；输出位于`_verification_output/`。

使用保存的逐目标预测核对均值与论文配对区间：

```bash
python -m pip install -r requirements-review.txt
python verify_predictions.py --bootstrap
```

预测文件含官方目标键、真实标签、折号、模型/种子及预测值；不加载模型或启动训练。主比较使用seed20261014，固定20轮消融使用seed20261007及折内牛只抽样种子，均为2,000次抽样。分数先在各折/种子计算，再按论文规则平均。

## 代码与数据位置

| 路径 | 内容 |
|---|---|
| `source/q3_sensor/` | 过去窗口、缺失表示、归一化、GRU模型及指标 |
| `source/scripts/` | 两类当前控制、统计特征、替代编码器、选型和汇总代码 |
| `source/configs/` | TS/OS划分、训练候选、固定配方和环境规格 |
| `results/all11_fast/` | 11方法比较的保存结果与配对区间 |
| `results/fixed20/` | 独立固定20轮组件实验；与主比较的Base分开 |
| `predictions/` | 310组主比较及240组正文消融case的紧凑预测包 |
| `evidence/` | 46,821目标、十牛支持数、五折角色与精确图例定位 |
| `figures/plot_data/` | 图2对齐和图5片段的CSV与来源记录 |

`source/`保留实验实际采用的Linux实现及历史路径，供核对方法；上面的审稿核查器可直接在本地运行。完整环境、原始观测、全部模型与缓存由作者的私人恢复库管理；不需要其访问权限即可运行本公开库的结果核查。

## 原始数据与许可

原始MmCows数据：[官方说明](https://github.com/neis-lab/mmcows)、[数据集页面](https://huggingface.co/datasets/neis-lab/mmcows)。本文所用40个成员的路径与SHA见`evidence/RAW_SOURCE_MEMBER_INDEX.json`。完整原始传感器CSV未在此重复上传。

复用观测、人工标签及其摘录遵循原数据的CC-BY-NC-SA-4.0许可；官方基准代码为MIT；MiniRocket源码保留GPL-3.0及原始归属。详见[来源与许可](docs/LICENSING.md)。本次未为所有新增内容另设统一许可证。

本稿为单日十牛资料上的探索性实证。公开核查验证的是保存预测/统计及其实现，不等同于重新训练或新环境验证。两套训练制度和随机种子在材料中分别标明。
