# MmCows传感器随机森林基准

本文的MmCows-RF使用三维位置、三个头部方向角和踝部派生躺卧状态共7维当前观测，随机森林100棵树、`random_state=42`、`class_weight='balanced'`。

## 实际采用的基准源码

源码位于[`source/reports/sensor_baselines_20261007/upstream/`](../source/reports/sensor_baselines_20261007/upstream/)，包含原始`data_loader.py`、`rf_classifer.py`、`train_uwb_hd_akl_s2.py`和`cmb_eval.py`。原作者为MmCows团队；固定提交为`78ba6d85abc41ef00eacebdd33a2704c08d110ab`，各文件原始URL与SHA256见同目录`provenance.json`，MIT许可全文见`LICENSE_MmCows.txt`。这些文件不作算法改写。

TS和OS划分配置分别位于[`official_ts.json`](../source/configs/official_ts.json)和[`official_os.json`](../source/configs/official_os.json)。本文正式比较与保存模型重放入口为[`extend_sensor_classic_original_rf_20261014.py`](../source/scripts/extend_sensor_classic_original_rf_20261014.py)，其中包括官方加载器调用、七维特征、正式目标键对齐与基线结果核对。

## 获取本文实际使用的数据

在仓库根目录运行：

```bash
python data/download_dataset.py
```

得到`data/extracted/sensor_data/`，含2023-07-25、C01—C10对应的40个原始CSV。来源、范围与许可见[`data/README.md`](../data/README.md)。

## 运行原作者的TS随机森林入口

使用Python 3.11，安装`numpy==1.24.4 pandas==2.0.3 scikit-learn==1.3.2 PyYAML pytz`。新建本地`path.yaml`，填写：

```yaml
sensor_data_dir: /absolute/path/to/Cattle_Behavior_Recognition_Paper/data/extracted/sensor_data
```

在仓库根目录运行（会训练五折随机森林并在源码下写入权重）：

```bash
python source/reports/sensor_baselines_20261007/upstream/train_uwb_hd_akl_s2.py --path_dir path.yaml --config_dir source/configs/official_ts.json
```

该入口直接调用随附原作者代码。OS加载接口为同一`data_loader.py`中的`data_loader_s1`，本文使用方式见正式比较入口的`author_frames`。完整历史模型的Linux训练入口、配置与环境版本位于`source/`；保存预测的跨平台核查方式见仓库首页。
