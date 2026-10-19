# 传感器七类行为开发基线

当前科学执行仅允许bitahub-root云端；以下早期CPU命令和“待正式评估”措辞保留为历史开发记录，不代表当前执行许可或状态。TS五折比较与21条件压力诊断已完成，结果分别见reports/sensor_cloud_formal_20261007和reports/sensor_cloud_stress_20261007。当前固定GRU实际为86,695参数；此前参数计数错误见reports/SENSOR_PARAMETER_COUNT_CORRECTION_20261007.md。

官方OS五折按configs/sensor_os_20261007.json运行：8头TRAIN＋VAL拟合、2头测试牛，标准化/权重/prior仅用拟合牛。OS适配与执行器位于scripts/sensor_os_protocol.py和run_sensor_formal_os.py，阶段状态见refine-logs/sensor_os_20261007。OS效果以完成后的独立报告为准。

固定牛1—10、官方fold1的TRAIN/VAL，不支持TEST读取或评分。三路输入为UWB三维位置（新协议厘米转米）、HD的roll/pitch/yaw圆周编码、Akl派生lying状态。

执行计划、冻结预算与same-family provisional代码审查在 `refine-logs/sensor_baselines_20261007/`。G0/G1没有实现，C0/C1只是普通拼接对照。

从项目根目录执行：

```powershell
python -m scripts.prepare_sensor_fold1 --stage metadata
python -m scripts.prepare_sensor_fold1 --stage extract
python -m scripts.build_sensor_development --targets-only
python -m scripts.build_sensor_development
python -m scripts.verify_sensor_implementation
python -m scripts.train_sensor_baselines
python -m scripts.summarize_sensor_baselines
```

默认source为 `data/sensor_fold1_20261007`，窗口缓存为 `data/sensor_windows_fold1_20261007`，结果为 `reports/sensor_baselines_20261007`。完整真实评分前应通过验证脚本；不要把合成CPU检查分数当科学结果。

数据名单独立于传感器覆盖，样本单位为牛号与固定15秒网格目标时刻。窗口 `(t-60,t]` 不跨连续split边界，同split相邻块合并；每秒表示只使用过去观测，保存有效掩码、当秒新观测、源时间和年龄。窗口开始不继承更早观测，低速输入前段等待不等于设备故障。

全部缺失目标保留，统一用TRAIN类别先验回退。标准化只用TRAIN真实有效观测，不能重复计入保持值。主指标固定七类宏F1；无标签类的分层宏F1也包含0项，须结合逐类support解释，不能仅凭桶间分数归因于陈旧性。

官方适配器保留作者的HD floor后算术平均、Akl nearest、标签0/NaN过滤及字段顺序。若最近Akl在授权范围外，只解析候选时间戳并排除对应可评分行，不读取其科学值，也不悄悄换更远来源。故当前为scoped开发特征复现，不能冒称发布的TRAIN＋VAL训练、五折TEST 73.7%。

远程ZIP索引探测可能收到范围外不透明字段字节，仅解释时间戳；没有完整ZIP SHA或完整成员CRC验证。记录时间戳过去限制不证明派生HD/Akl严格实时可用。

训练默认CPU、4线程、batch256、20epoch、seeds17/23/29、balanced CE、AdamW(.001,.01)、无增强/调度。已完成同身份运行会跳过，未完成运行按同seed从epoch1重算，不是epoch级续训。每个神经运行保存最佳VAL宏F1的最早checkpoint、全部20轮历史和逐目标预测。
