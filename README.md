
## 1\. 架构


如图所示，IADBE（Industrial Anomaly Detection Benchmark Engine）系统由三个主要部分组成： IADBE、IADBE 服务器和 IADBE 后台。IADBE 是系统的核心，API 和 CLI 是网关。数据集、模型和指标是系统最重要的部分。模型基于开源的 Anomalib 和 YOLOv8。系统有三个主要入口： 训练、测试和预测。
![image](https://img2024.cnblogs.com/blog/1201453/202410/1201453-20241027191325751-1813444188.png)
在 IADBE 服务器中，我们使用了 SpringBoot 和 Kotlin 框架。IADBE 服务器从 IADBE 获取训练和测试结果文件，然后进行处理，包括日志和 CSV 处理。在 IADBE 前端，我们使用了 Angular 框架。前端向服务器发送 HTTP 请求，然后服务器将 JSON 数据发回前端。前端提供既时尚又实用的可视化效果。
为了简化部署过程，我们使用Docker Compose将 IADBE、IADBE 服务器和 IADBE 前端镜像放在一起，然后一起运行。我们使用 GitHub 进行版本控制。我们在 GitHub 上发布所有相关软件，包括IADBE、IADBE Server 和IADBE Frontend，以便复制和验证。我们已将自定义数据 集和预训练模型上传到 Huggingface。我们在一系列操作系统上测试了跨平台功能，包括 Windows 11 和 10，以及 Ubuntu 22 和 20。结果表明，这些操作系统在运行 IADBE 系统没有问题。


## 2\. 介绍


值得注意的是，以往伴随开源项目进行的研究往往会因为环境部署问题而耗费大量时间，有些项目还会因为版本冲突问题而无法运行。不过，在本项目中，解决方案将使用 Docker 或 Conda 进行部署。该项目的主要目标是为研究人员提供一个开箱即用的工业异常检测平台。该平台主要基于Anomalib，Ultralytics，应能重现和识别以前的研究，尽量避免bug且易于部署。


## 3\. 安装


已在 Linux (Ubuntu22/20\)、Windows (Win11/10\) 上测试 ✅


IADBE 提供两种安装库的方式： Conda 和 Docker。如果想修改依赖包和其版本并在开发模式下工作，请使用 Conda。如果想完全复制我们的环境（python、torrent......），请使用 Docker。⚠️假设您已安装 nvidia 驱动程序和 CUDA。否则，您可以使用 CPU 进行训练。


### 3\.1 Conda安装


从Conda安装依赖



```
# Use of virtual environment is highly recommended
# Using conda
conda create -n IADBE python=3.10
conda activate IADBE

# To avoid anomalib install bug
pip install pip==24.0

# Install anomalib
pip install anomalib==1.1.0

# Install the full package, this will install Anomalib CLI. Anomalib CLI is a command line interface for training, testing.
anomalib install

# Install ultralytics
pip install ultralytics

# Or using your favorite virtual environment
# ...

```

### 3\.2 Docker安装


从Docker安装依赖



```
# Clone the repository and install in editable mode
git clone https://github.com/cjy513203427/IADBE.git
cd IADBE

# Build docker image
docker build --no-cache -t iadbe .
# Run docker container
docker run --gpus all -it --shm-size="8g" --name iadbe --rm iadbe bash

```

您可以将其作为虚拟机，使用相同的命令进行训练、测试和推理，也可以将 docker env设置为外部环境。


## 4\. 数据集


### 4\.1 标准数据集


IADBE 可通过 API/CLI 自动下载标准数据集（MVTec、MVTec3D、Btech、VisA、Kolektor）。如果您在下载数据集时遇到任何问题，也可以直接从其官方网站下载。


### 4\.2 自定义数据集


我们上传了一个自定义数据集到[IADBE\_Custom\_Dataset](https://github.com). 它包含 anomalib 和 YOLO 格式的数据集。您可以用 Huggingface 方法导入，或者直接从 GitHub 克隆并下载到本地。


#### 4\.2\.1 使用huggingface



```
from datasets import load_dataset
ds = load_dataset("gt111lk/IADBE_Custom_Dataset")

```

#### 4\.2\.2 Git Clone



```
git clone https://huggingface.co/datasets/gt111lk/IADBE_Custom_Dataset

```

## 5\. 训练和测试


IADBE 支持基于 API 和 CLI 的训练。API 更为灵活，允许更多定制，而 CLI 训练则使用命令行界面，对于那些想快速上手使用 IADBE 的人来说可能更容易。


### 5\.1 使用API进行训练和测试


train\_test\_mvtec\_xxx.py 文件看起来像这样。使用集成开发环境或python train\_test\_mvtec\_xxx.py运行该文件，即可开始使用整个 MVTec 数据集进行训练。



```
import logging
from anomalib import TaskType
from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import Padim

# configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

datasets = ['screw', 'pill', 'capsule', 'carpet', 'grid', 'tile', 'wood', 'zipper', 'cable', 'toothbrush', 'transistor',
            'metal_nut', 'bottle', 'hazelnut', 'leather']

for dataset in datasets:
    logger.info(f"================== Processing dataset: {dataset} ==================")
    model = Padim()
    datamodule = MVTec(category=dataset, num_workers=0, train_batch_size=256,
                       eval_batch_size=256)
    engine = Engine(pixel_metrics=["AUROC", "PRO"], image_metrics=["AUROC", "PRO"], task=TaskType.SEGMENTATION)

    logger.info(f"================== Start training for dataset: {dataset} ==================")
    engine.fit(model=model, datamodule=datamodule)

    logger.info(f"================== Start testing for dataset: {dataset} ==================")
    test_results = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
    )

```

### 5\.2 使用CLI进项训练和测试


train\_test\_mvtec\_xxx.sh 文件看起来像这样。使用bash train\_test\_mvtec\_xxx.sh运行该文件，即可使用整个 MVTec 数据集启动训练。



```
#!/bin/bash

datasets=('screw' 'pill' 'capsule' 'carpet' 'grid' 'tile' 'wood' 'zipper' 'cable' 'toothbrush' 'transistor' 'metal_nut' 'bottle' 'hazelnut' 'leather')
config_file="./configs/models/padim.yaml"

for dataset in "${datasets[@]}"
do
    command="anomalib train --data anomalib.data.MVTec --data.category $dataset --config $config_file"
    echo "Running command: $command"
    # Excute command
    $command
done

```

要深入了解 Anomalib CLI 时，可以检索 [Training via CLI from Training](https://github.com)


## 6\. 推理


Anomalib 包含多个推理脚本，包括 Torch、Lightning、Gradio 和 OpenVINO 推理器，可使用训练/导出的模型执行推理。这里我们展示一个使用 Lightning 推断器进行推理的示例。如果您想在不进行训练的情况下测试我们的预训练模型，可以在 [HuggingfaceIADBE\_Models](https://github.com "HuggingfaceIADBE_Models") 上找到它。


### 6\.1 使用API推理


下面的示例演示了如何通过从检查点文件加载模型来执行 Lightning 推理。



```
# Assuming the datamodule, custom_model and engine is initialized from the previous step,
# a prediction via a checkpoint file can be performed as follows:
predictions = engine.predict(
    datamodule=datamodule,
    model=model,
    ckpt_path="path/to/checkpoint.ckpt",
)

```

### 6\.2 使用CLI推理



```
# To get help about the arguments, run:
anomalib predict -h

# Predict by using the default values.
anomalib predict --model anomalib.models.Patchcore \
                 --data anomalib.data.MVTec \
                 --ckpt_path 

# Predict by overriding arguments.
anomalib predict --model anomalib.models.Patchcore \
                 --data anomalib.data.MVTec \
                 --ckpt_path 
                 --return_predictions

# Predict by using a config file.
anomalib predict --config  --return_predictions

```

## 7\. 自定义数据集模式


IADBE 可以帮助您对自定义数据集进行训练和推理。默认数据集格式基于 Anomalib，但您也可以使用 YOLO。只需查看[yolo\_custom\_dataset\_setting](https://github.com).


1. 首先要做的是将自己的数据集导入到项目中，并创建自定义数据配置文件。


仅有正常图像的配置



```
class_path: anomalib.data.Folder
init_args:
  name: "custom_dataset"
  root: "datasets/Custom_Dataset/hazelnut"
  normal_dir: "train/good"
  abnormal_dir: "test/crack"
  mask_dir: null
  normal_split_ratio: 0.2
  test_split_mode: synthetic

```

有正常和异常图像的配置



```
class_path: anomalib.data.Folder
init_args:
  name: "custom_dataset"
  root: "datasets/Custom_Dataset/chest_xray"
  normal_dir: "train/good"
  abnormal_dir: "test/crack"
  normal_test_dir: "test/good"
  normal_split_ratio: 0
  extensions: [".png"]
  image_size: [256, 256]
  train_batch_size: 32
  eval_batch_size: 32
  num_workers: 8
  task: classification
  train_transform: null
  eval_transform: null
  test_split_mode: synthetic
  test_split_ratio: 0.2
  val_split_mode: same_as_test
  val_split_ratio: 0.5
  seed: null

```

2. 接下来，您需要选择一个模型，然后使用自定义数据集对其进行训练。



```
anomalib train --data  --model anomalib.models.Padim 

```

3. 最后，您可以使用预训练的模型运行推理，以获得预测结果。



```
anomalib predict --model anomalib.models.Padim \
                 --data  \
                 --ckpt_path 

```

## 8\. 基准测试结果


这些是基准测试的部分结果。您可以在我的论文中找到更多细节，请参见 GitHub 容器中文件夹`papers/` 。这些结果来自原始日志，您可以在`logs/rawlogs` 下找到。这些模型在 Ubuntu 22\.04 和 RTX 3090 上进行了评估。




|  | Screw | Pill | Capsule | Carpet | Grid | Tile | Wood | Zipper | Cable | Toothbrush | Transistor | Metal Nut | Bottle | Hazelnut | Leather | Average | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CFA | 87\.68 | 93\.10 | 48\.17 | 50\.00 | 84\.04 | 37\.50 | 62\.50 | 40\.34 | 61\.96 | **98\.06** | 45\.00 | 39\.78 | 50\.00 | 60\.00 | 50\.00 | 60\.54 | **99\.30** |
| CFLOW | 81\.57 | 92\.91 | **95\.45** | **97\.27** | 88\.47 | **100\.00** | **99\.39** | **97\.64** | 94\.15 | 94\.17 | 94\.88 | **100\.00** | **100\.00** | **99\.68** | **100\.00** | **95\.71** | **96\.31** |
| CSFLOW | 41\.58 | 38\.09 | 60\.77 | **99\.16** | 78\.36 | 90\.44 | **95\.75** | 88\.93 | 67\.31 | 44\.72 | 50\.54 | 74\.46 | 89\.92 | 71\.02 | **100\.00** | 72\.74 | 93\.50 |
| DFKDE | 69\.17 | 64\.18 | 72\.04 | 68\.42 | 46\.69 | 92\.50 | 81\.32 | 88\.26 | 68\.67 | 78\.89 | 81\.12 | 76\.20 | 92\.94 | 77\.57 | 77\.79 | 75\.72 | 77\.40 |
| DFM | 77\.79 | **96\.89** | 92\.98 | 90\.93 | 65\.41 | **98\.67** | **97\.81** | **97\.51** | 94\.27 | **96\.39** | 94\.79 | 91\.72 | **100\.00** | **96\.82** | **100\.00** | 92\.80 | 94\.30 |
| DRAEM | 30\.01 | 74\.55 | 77\.02 | 72\.47 | 80\.45 | 86\.80 | **95\.96** | 78\.90 | 63\.98 | 70\.42 | 90\.21 | 93\.55 | **98\.10** | 77\.50 | 86\.65 | 78\.44 | **98\.00** |
| DSR | 56\.67 | 70\.16 | 72\.72 | 43\.82 | **97\.08** | 80\.70 | 90\.75 | 78\.97 | 76\.96 | **96\.94** | 91\.04 | 81\.04 | 86\.59 | 81\.64 | 83\.97 | 79\.27 | **98\.20** |
| FASTFLOW | 65\.85 | 76\.60 | 69\.64 | 93\.38 | **96\.32** | 93\.36 | **98\.16** | 72\.69 | 67\.62 | 72\.50 | 89\.62 | 81\.33 | **99\.68** | 79\.86 | **99\.90** | 83\.77 | **99\.40** |
| FRE | 58\.11 | 83\.82 | 76\.59 | 94\.18 | 64\.75 | **98\.99** | **98\.16** | 93\.91 | 85\.53 | **95\.00** | 90\.92 | 84\.12 | **99\.37** | 92\.29 | **99\.93** | 87\.71 | **98\.40** |
| GANomaly | 32\.79 | 59\.96 | 26\.57 | 21\.71 | 57\.23 | 54\.42 | 60\.88 | 41\.05 | 52\.47 | 49\.17 | 33\.46 | 26\.30 | 47\.78 | 53\.86 | 36\.68 | 43\.62 | 42\.10 |
| PaDiM | 78\.95 | 79\.95 | 86\.48 | **97\.99** | 87\.47 | 94\.55 | **97\.46** | 77\.46 | 85\.96 | 82\.50 | 94\.54 | **98\.34** | **99\.52** | 88\.32 | **100\.00** | 89\.97 | **96\.70** |
| PatchCore | **98\.11** | 94\.76 | **97\.85** | **99\.12** | **98\.08** | **98\.81** | **98\.77** | **99\.21** | **99\.10** | **100\.00** | **100\.00** | **99\.80** | **100\.00** | **100\.00** | **100\.00** | **98\.91** | **98\.10** |
| RD | **98\.03** | **97\.63** | **97\.93** | **99\.36** | **95\.49** | **100\.00** | **99\.39** | **97\.16** | **95\.45** | 91\.39 | **97\.87** | **100\.00** | **100\.00** | **100\.00** | **100\.00** | **97\.98** | **98\.50** |
| RKDE | 50\.58 | 68\.77 | 51\.93 | \- | 75\.36 | 67\.72 | 62\.54 | 75\.37 | 85\.83 | 77\.17 | 65\.00 | 90\.63 | 85\.07 | **100\.00** | **100\.00** | 75\.43 | \- |
| STFPM | 77\.62 | 40\.78 | 60\.59 | **98\.88** | 59\.23 | **97\.08** | **98\.95** | 75\.39 | 91\.34 | 47\.50 | 61\.88 | 40\.22 | 43\.97 | **96\.50** | **100\.00** | 72\.66 | **97\.00** |
| UFLOW | 49\.19 | 94\.84 | 56\.72 | **100\.00** | **99\.33** | **99\.39** | **95\.09** | 89\.73 | 62\.67 | 64\.17 | 81\.46 | 55\.77 | **99\.21** | 90\.39 | **100\.00** | 82\.53 | **98\.74** |


这些是有关 Image AUROC 的基准结果。最后一栏是源论文作者声明的结果。CFA的既定结果（99\.30）与我们评估得出的平均结果（60\.54）之间存在明显差异。不过，CSFLOW、DREAM、DSR、FASTFLOW、FRE、PaDiM、STFPM 和 UFLOW 的结果与我们的评估结果差距相对较小。相比之下，CFLOW、DFM、PatchCore 和 RD 与我们的结果完全一致。


## 9\. 相关链接


* [IADBE Core](https://github.com)
* [IADBE Server](https://github.com):[milou加速器](https://jiechuangmoxing.com)
* [IADBE Frontend](https://github.com)
* [IADBE Pre\-trained Models](https://github.com)
* [IADBE Custom Dataset](https://github.com)
* [Inference with Gradio deployed on HF spaces](https://github.com)
* [Final Version Paper](https://github.com)


## 10\. 未来工作


扩展与 Gradio 的用户交互功能
扩展与 Huggingface 的集成
使用 Streamlit 简化前台和后台
美化基准测试的可视化效果
实施 IADBE 系统管道
多类统一的模型基准测试
基于 Anomalib 复现高性能模型
基于 Anomalib 增强对视频数据集的支持


## 11\. 杂谈


我即将在达姆施塔特工业大学完成计算机专业的硕士论文，真不容易！欢迎学弟学妹们和工业界的朋友使用和基于我的 跨平台/语言 项目进行二次开发，希望它能为你们提供一些灵感。


同时，热烈欢迎加入 IADBE 开源项目，共同推进工业异常检测基准引擎的扩展。遇到问题可以提交 Issues，想贡献的也欢迎通过 Pull Requests 一起参与开发！


