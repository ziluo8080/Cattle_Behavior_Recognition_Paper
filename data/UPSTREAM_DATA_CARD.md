---
license: cc-by-nc-sa-4.0
viewer: false
annotations_creators:
- expert-generated
language: []
language_creators:
- expert-generated
multilinguality: []
pretty_name: mmcows
size_categories:
- n>1T
source_datasets:
- original
tags:
- cattle
- cows
- cow
- animals
- multimodal
- visual localization
- sensor fusion
- UWB localization
- precision livestock farming
task_categories:
- image-classification
- object-detection
task_ids:
- multi-class-image-classification
---

# MmCows: A Multimodal Dataset for Dairy Cattle Monitoring

<!-- MmCows is a large-scale multimodal dataset for behavior monitoring, health management, and dietary management of dairy cattle.

The dataset consists of data from 16 dairy cows collected during a 14-day real-world deployment, divided into two modality groups. The primary group includes 3D UWB location, cows' neck IMMU acceleration, air pressure, cows' CBT, ankle acceleration, multi-view RGB images, indoor THI, outdoor weather, and milk yield. The secondary group contains measured UWB distances, cows' head direction, lying behavior, and health records.

MmCows also contains 20,000 isometric-view images from multiple camera views in one day that are annotated with cows' ID and their behavior as the ground truth. The annotated cow IDs from multi-views are used to derive their 3D body location ground truth.
 -->

Details of the dataset and benchmarks are available [here](https://github.com/neis-lab/mmcows).

For a quick overview of the dataset, please check this [video](https://www.youtube.com/watch?v=YBDvz-HoLWg).

<br />

# Instruction for downloading

### 1. Install requirements
```bash
pip install huggingface_hub
```
See the file structure [here](https://huggingface.co/datasets/neis-lab/mmcows/tree/main) for the next step.

### 2. Download a file individually
To download ```visual_data.zip``` to your ```local-dir```, use command line:
```bash
huggingface-cli download \
  neis-lab/mmcows \
    visual_data.zip \
  --repo-type dataset \
  --local-dir ./
```
Using a Python script:
```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="neis-lab/mmcows",
    repo_type="dataset",
    local_dir="./",
    filename="visual_data.zip"
)
```
To download other files, replace ```visual_data.zip``` with the file you want to download.

### 3. Download all files inside a folder
To download all files in ```1s_interval_images_3hr``` to your ```local-dir```, use command line:
```bash
huggingface-cli download \
  neis-lab/mmcows \
  --repo-type dataset \
  --include "1s_interval_images_3hr/*" \
  --local-dir ./
```