# 论文与审稿资料对应关系

| 论文内容 | 可核查资料 |
|---|---|
| 数据、类别与逐牛分布 | `evidence/CLASS_SUPPORT.csv`、`COW_CLASS_SUPPORT.csv`、`FINAL_46821_TARGET_KEYS.jsonl.gz` |
| TS/OS五折定义 | `source/configs/official_ts.json`、`official_os.json`、`evidence/OFFICIAL_ROLE_RANGES.csv`、`ALL_FOLD_TARGET_ROLE_INDEX.jsonl.gz` |
| 26维输入与过去对齐 | `source/q3_sensor/protocol.py` |
| 图3及GRU实现 | `source/q3_sensor/endpoint_models.py` |
| 当前完整端点重复控制 | `source/scripts/sensor_history_control_models_20261011.py` |
| 110维特征与统计RF | `source/scripts/sensor_classic_features_20261012.py`；补充表S7 |
| 表3、表4、表6—8及补充表S1 | `results/all11_fast/`与`predictions/main_TS.npz`、`main_OS.npz` |
| 表5 | `results/fixed20/key_ablations.csv`及`predictions/fixed20_*.npz` |
| 补充表S2结构/参数量 | `source/scripts/sensor_architecture_models_20261011.py`、`sensor_tcn_model_fast_20261014.py` |
| 图2/图5目标键 | `evidence/FIGURE2_EXACT_TARGET_KEY.json`、`FIGURE5_EXACT_TARGET_KEYS.csv`及`figures/plot_data/` |

主文保留8张主要表格；补充表S1—S7集中于独立Word。JSONL采用gzip无损压缩，核查器自动读取。模型ID与文稿中文名见各结果CSV及实际选型锁定文件`results/recipe_locks/`。

逐目标预测从历史CSV无损转换为NPZ：`prediction`每行对应`model`/`seed`，每列对应`cow_id`/`timestamp`/`label`/`fold`；NPZ使用非对象数组，读取时设置`allow_pickle=False`。来源文件SHA与转换记录见`predictions/PREDICTION_EXPORT.json`。
