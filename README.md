# 奶牛行为识别论文审稿资料

对应中文稿《融合多源短时历史信息与当前状态的奶牛行为识别》，v10。此仓库提供本研究的数据子集、MmCows随机森林基准与方法代码、论文材料、保存预测及图表来源。

固定版本标签：`paper-v10-20261019-r2`。本仓库不是新增数据采集，原始数据来自MmCows。

## 数据集与基准代码下载

本文使用的数据子集包含2023-07-25、C01—C10的40个原始CSV，未压缩395,421,225字节，压缩包114,340,262字节。压缩包位于本仓库固定Release：

- [直接下载数据子集](https://github.com/ziluo8080/Cattle_Behavior_Recognition_Paper/releases/download/paper-v10-20261019-r2/mmcows_0725_c01_c10_sensor_subset.tar.gz)
- [数据说明、来源与许可](data/README.md)
- [MmCows随机森林基准源码与运行步骤](benchmarks/README.md)

在仓库根目录运行以下标准库脚本，可下载数据并校验全部40个CSV：

```bash
python data/download_dataset.py
```

数据子集沿用MmCows的归属与CC-BY-NC-SA-4.0许可，基准源码保留MIT许可。此处公开的是本研究实际使用的子集，完整MmCows数据仍见原作者项目。

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
| `data/` | 40个原始CSV的Release下载、校验与来源 |
| `benchmarks/` | 原作者随机森林源码导航及运行步骤 |
| `source/q3_sensor/` | 过去窗口、缺失表示、归一化、GRU模型及指标 |
| `source/scripts/` | 两类当前控制、统计特征、替代编码器、选型和汇总代码 |
| `source/configs/` | TS/OS划分、训练候选、固定配方和环境规格 |
| `results/all11_fast/` | 11方法比较的保存结果与配对区间 |
| `results/fixed20/` | 独立固定20轮组件实验；与主比较的Base分开 |
| `predictions/` | 310组主比较及240组正文消融case的紧凑预测包 |
| `evidence/` | 46,821目标、十牛支持数、五折角色与精确图例定位 |
| `figures/plot_data/` | 图2对齐和图5片段的CSV与来源记录 |

`source/`保留实验实际采用的Linux实现及历史路径，供核对方法；上面的审稿核查器可直接在本地运行。本研究实际使用的原始观测子集已在公开Release提供；数据下载与校验见`data/`，MmCows随机森林基准入口见`benchmarks/`。全部历史模型、缓存及完整环境另有作者工作区备份。

## 原始数据与许可

原始MmCows数据：[官方说明](https://github.com/neis-lab/mmcows)、[数据集页面](https://huggingface.co/datasets/neis-lab/mmcows)。本文所用40个成员的路径与SHA见`evidence/RAW_SOURCE_MEMBER_INDEX.json`。本文实际使用的40个原始CSV已作为本仓库固定Release的数据附件公开，原始CSV内容逐字节保留。

复用观测、人工标签及其摘录遵循原数据的CC-BY-NC-SA-4.0许可；官方基准代码为MIT；MiniRocket源码保留GPL-3.0及原始归属。详见[来源与许可](docs/LICENSING.md)。本次未为所有新增内容另设统一许可证。

本稿为单日十牛资料上的探索性实证。公开核查验证的是保存预测/统计及其实现，不等同于重新训练或新环境验证。两套训练制度和随机种子在材料中分别标明。
