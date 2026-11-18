# v48 论文数据与核心代码

《融合多尺度时序特征与统计特征的奶牛行为识别方法》配套材料。论文在 MmCows 2023 年 7 月 25 日的 10 头奶牛数据上开展时间划分与个体划分五折评价。TT-CSF 的宏平均 F1 分别为 86.765% 和 84.016%。

- [原始数据下载](https://github.com/ziluo8080/Cattle_Behavior_Recognition_Paper/releases/download/data-code-20261030-v29/mmcows_0725_c01_c10_50csv.tar.gz)。包含 50 份原始 CSV，覆盖原生 IMMU、UWB、头部方向、踝部状态和行为标签。归档来源、SHA256 和下载校验方式见 [data](../data/README.md)。数据归属 Hien Vu 与 MmCows 作者，原始发布版本标注 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)，详见[上游数据卡](https://huggingface.co/datasets/neis-lab/mmcows/blob/532a193d5fd90ed6cfb2d640923c4358ceca00b4/README.md)。软件许可适用于相应代码。
- [核心代码](source/)和[文件校验表](CODE_SHA256.json)。包括双尺度数据构建、ConvTran 末端残差交互、统计特征、统一选模、融合、消融与配置。锁定训练工件以原工程相对路径归档，原训练代码依赖对应的环境和输入缓存。仓库根目录提供原始数据获取和基础依赖入口。
- [复算附件](https://github.com/ziluo8080/Cattle_Behavior_Recognition_Paper/releases/tag/paper-v48-data-code-20261118)含冻结概率及对应目标；[results](results/)记录逐折分类结果，[calibration](calibration/)保留校准轮次记录。表5各基线的旧版冻结验证入口见[仓库根目录](../README.md)。

## 复算当前结果

Python 3.10+ 和 NumPy。在仓库根目录执行：

```bash
python paper_v48/verify.py --archive /path/to/v48_frozen_predictions.tar.gz --output recount_v48
```

脚本核对附件和成员 SHA256，重新计算 840 个混淆矩阵及逐类 F1，并按折内种子平均和五折等权汇总。训练及预测由原实验完成，复算使用这些已保存的概率和真实标签。
