# MmCows单日十牛数据子集

50个原始CSV：10个原生IMMU，另有位置、头部方向、踝部状态和行为标签各10个。
CSV与原始发布成员逐字节一致，未重新采集或改写。下载及逐文件校验：

```bash
python run.py --task data
python run.py --task data --verify-only
```

归档在本公开仓库Release，具体地址和SHA256见DATASET_ASSET.json。解压至data/extracted/sensor_data。
TS/OS五折定义见configs/official_ts.json与official_os.json。目标为2023年7月25日C01–C10七类行为，正式目标集合46821行。
上游数据归属Hien Vu及MmCows作者，项目https://github.com/neis-lab/mmcows；原论文NeurIPS2024，DOI10.52202/079017-1898。
指定HuggingFace版本数据卡标示CC BY-NC-SA4.0，论文补充标示CC BY-NC4.0。两种声明均保留，按上游适用许可使用；本仓库软件许可不覆盖第三方数据。
