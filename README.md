# TT-CSF：奶牛多源传感器行为识别

[论文 v48 的原始数据、核心代码与结果复算](paper_v48/README.md)

本公开仓库保存数据集子集与实验代码。当前方法为双时间尺度ConvTran-ERI与XGBoost-866的概率间隔加权融合。
历史v10材料保留在Git历史及旧Release；当前默认分支按模型仓库方式组织，不收录论文、绘图或写作工作材料。

## 目录

| 路径 | 内容 |
|---|---|
| data/ | 50个原始CSV的公开下载、来源与哈希；数据不依赖私有仓库 |
| configs/ | 官方TS/OS划分、主模型及对照的实际固定配置 |
| models/ | 主模型入口；实际实现位于scripts/与experiments/ |
| data_provider/ | 最近历史对齐、缺失表示、标准化、特征及数据准备索引 |
| scripts/ | 实际数据处理、校准、训练、重训、预测和评分实现 |
| q3_sensor/ | 时间窗口、输入表示与评价基础实现 |
| experiments/ | GRU、Transformer、ModernTCN、TimesNet及MultiRocket的实际适配实现和必要第三方源代码 |
| third_party/ | 受许可复用的源代码，保留原作者与许可证 |
| benchmarks/ | 当前五折结果的混淆计数及期望F1，供数值复核 |
| tools/ | 数据下载校验与结果复算工具 |

目录组织参考Time-Series-Library与ConvTran公开仓库；本文方法与其任务/结构均有区别。

## 快速核查

Python3.11。数据下载和结果复算只用标准库，不启动训练：

```bash
python run.py --task inspect
python run.py --task verify-results
python run.py --task data
```

安装训练依赖后检查主模型的前向接口（随机初始化，只检查形状，不是实验准确率）：

```bash
python -m pip install -r requirements.txt
python run.py --task model-check
```

## 数据处理与训练入口

- 原生动态处理：scripts/sensor_native_immu_data_20261021.py，保留100点10秒窗口。
- 状态/上下文：q3_sensor/protocol.py、scripts/sensor_motion_pair.py与相关准备脚本。
- 五折输入准备：scripts/prepare_shared_fusion_fivefold_20261024.py与prepare_converged_fivefold_20261027.py。
- 主模型调度：scripts/run_converged_fivefold_20261027.py。
- GRU/Transformer/ModernTCN/MultiRocket：scripts/run_comparator_fivefold_retrain_20261029_v14.py。
- TimesNet：scripts/run_timesnet_fivefold_20261029_v20.py。
- 神经训练核、树拟合、冻结预测与最终评分分别保留在各入口导入的实际模块中。

公开代码保留真实实验模块名、配置及固定Linux路径/q3/mmcows-sensor-behavior-20261007，便于核对实现。
原全流程调度依赖多阶段缓存和冻结证据，不是仅克隆后即一条命令完成全部历史实验的便携训练器。
可独立下载原始数据、检查模型接口及复算结果；完整已训练状态、缓存和恢复工具由作者私有实验库管理。
本次归档验证不启动训练，不把前向接口测试或结果复算称为全环境重训验证。

## 当前固定配置五折结果

三种子17/23/29先在折内平均F1，再汇总五折均值与样本标准差。
TT-CSF的TS宏F1为86.801±0.937%，OS为84.134±1.944%；完整逐类结果由verify-results输出。
这是同一天十头牛、具有既往开发与测试暴露的固定配置复评，不是新的独立盲测。
TS/OS并列评价。历史GRU不在当前对照，保留GRU-DS；TimesNet-DS替代TSLANet-DS。

## 归属与许可

本仓库保留MmCows、ConvTran、TimesNet、ModernTCN、MultiRocket等上游作者归属和代码许可证。
原数据按其上游数据许可使用；第三方许可证优先于本仓库软件LICENSE，不能把软件MIT授权延伸到第三方数据或代码。
