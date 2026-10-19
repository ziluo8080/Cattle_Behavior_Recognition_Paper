# 本文使用的MmCows传感器数据子集

本目录提供本文实际使用的40个原始CSV的GitHub下载入口与逐文件校验。数据来自MmCows原作者公开发布的2023-07-25传感器与行为标签，包含C01—C10十头奶牛对应的位置、头部方向、踝部派生状态和人工行为标签。这里发布的是原数据的研究子集，未重新采集或改写CSV内容。

## 下载和校验

```bash
python data/download_dataset.py
```

脚本从本仓库固定版本的GitHub Release下载压缩包，校验压缩包与40个CSV的SHA256，解压到`data/extracted/sensor_data/`。直接下载链接、字节数与SHA256见[`DATASET_ASSET.json`](DATASET_ASSET.json)。压缩包作为Release附件保存，未放入Git文件历史。

已经下载压缩包时可运行：

```bash
python data/download_dataset.py --archive /path/to/mmcows_0725_c01_c10_sensor_subset.tar.gz
python data/download_dataset.py --archive /path/to/mmcows_0725_c01_c10_sensor_subset.tar.gz --verify-only
```

逐文件来源路径、原始字节数和SHA256见[`RAW_SOURCE_MEMBER_INDEX.json`](../evidence/RAW_SOURCE_MEMBER_INDEX.json)。此子集未包含MmCows两周的全部传感器记录或图像数据；完整数据仍由原作者渠道提供。

## 原始来源与归属

- 原始数据集：[MmCows官方项目](https://github.com/neis-lab/mmcows)
- 官方数据托管：[Hugging Face](https://huggingface.co/datasets/neis-lab/mmcows)
- 采用的原始归档修订：`532a193d5fd90ed6cfb2d640923c4358ceca00b4`，`sensor_data.zip`
- 原作者：Hien Vu、Omkar Prabhune、Unmesh Raskar、Dimuth Panditharatne、Hanwook Chung、Christopher Y. Choi、Younghyun Kim。
- 原论文：MmCows: A Multimodal Dataset for Dairy Cattle Monitoring，NeurIPS 2024 Datasets and Benchmarks Track；DOI：[10.52202/079017-1898](https://doi.org/10.52202/079017-1898)。

## 许可

依据官方Hugging Face数据页当前声明，此数据子集遵循[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)，保留原作者归属与来源，按原许可使用。CSV内容与研究采用的原文件逐字节一致；仅重新打包为压缩子集。第三方基准软件适用其MIT许可，二者分别标明，详见[`docs/LICENSING.md`](../docs/LICENSING.md)。
