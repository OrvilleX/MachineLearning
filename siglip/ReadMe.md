# SigLIP 图文对照模型

CLIP模型是使用softmax（交叉熵损失）训练，其中softmax需要对所有成对相似性有一个全局视图，以便归一化到概率。经研究发现，可用更简单的 sigmoid 损失来代替它，其不需要这种全局视图。sigmoid 损失同时允许在预训练期间进一步扩大批量大小，在较小的批量尺寸下也表现得更好。该模型在零样本图像分类和图像文本检索方面都优于CLIP模型，比如下图所示：
![](/media/17378753829365.jpg)

参考文档
* [SigLIP模型的推理demo](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SigLIP/Inference_with_(multilingual)_SigLIP%2C_a_better_CLIP_model.ipynb)  
* [SigLIP官方使用文档](https://huggingface.co/docs/transformers/main/model_doc/siglip#)  
* [siglip-so400m-patch14-384目前最优模型](https://huggingface.co/google/siglip-so400m-patch14-384)  
* [文档中相关代码仓库](https://github.com/OrvilleX/Apollo-LLM-Start/tree/main/siglip)

推理系统环境
* CPU: 12 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz
* MEM: 90GB
* GPU: RTX 4090(24GB) * 1
* 系统: ubuntu22.04
* 环境: PyTorch  2.5.1 + Python 3.12 + Cuda  12.4

## 一、快速使用

### 1. 基本配置

```bash
# 将pip跟conda依赖包安装到数据盘
mkdir -p /root/autodl-tmp/conda/pkgs
conda config --add pkgs_dirs /root/autodl-tmp/conda/pkgs

mkdir -p /root/autodl-tmp/conda/envs
conda config --add envs_dirs /root/autodl-tmp/conda/envs
```

```bash
# 创建conda环境
conda create -y -n siglip python=3.10 -q
conda init bash && source /root/.bashrc

# 安装依赖库
conda activate siglip
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# 配置到jupyterlab
conda install ipykernel
ipython kernel install --user --name=siglip
```

### 2. 项目配置

```bash
# 安装git lfs下载镜像
sudo apt update
sudo apt install nload git-lfs -y

# 安装项目依赖
conda activate siglip
pip install transformers==4.44.0 sentencepiece protobuf jupyterlab_widgets

# 离线下载模型
git lfs install
git clone https://huggingface.co/google/siglip-so400m-patch14-384
```

### 3. 快速应用

接下来将基于Jupyter lab进行基本使用的方式介绍，首先是加载目前最好的模型，它具有
shape-optimized（so）架构，该模型的性能明显优于 ViT 巨型架构，同时体积要小得多。

```python
from transformers import AutoProcessor, AutoModel

# 下方离线模型的路径需要根据实际的路基进行调整
processor = AutoProcessor.from_pretrained("/root/autodl-tmp/siglip-so400m-patch14-384")
model = AutoModel.from_pretrained("/root/autodl-tmp/siglip-so400m-patch14-384")
```

准备需要识别图片，通过网络进行下载并加载。

```python
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image
```

下面准备对应的需要进行测试验证的文字与图片进行处理。

```python
texts = ["a photo of 2 cats", "a photo of 2 hamburgers", "a photo of 2 dogs"]

inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")
for k,v in inputs.items():
  print(k,v.shape)
```

接下来，我们执行前向传递并获取每个图像-文本对的未规范化分数，在应用sigmoid激活函数，转换为单个概率。

```python
import torch

with torch.no_grad():
  outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image) # these are the probabilities
print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
print(f"{probs[0][1]:.1%} that image 0 is '{texts[1]}'")
print(f"{probs[0][2]:.1%} that image 0 is '{texts[2]}'")
```

上述的流程包含了较多的前处理与后处理，为了更简便的进行分析处理，可以通过管道(pipelines)简化这一步骤。

```python
from transformers import pipeline
from PIL import Image
import requests
import torch

device = 0 if torch.cuda.is_available() else -1
# load pipe
image_classifier = pipeline(task="zero-shot-image-classification", model="/root/autodl-tmp/siglip-so400m-patch14-384", device=device)

# load image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# inference
outputs = image_classifier(image, candidate_labels=["2 cats", "a plane", "a remote"])
outputs = [{"score": round(output["score"], 4), "label": output["label"] } for output in outputs]
print(outputs)
```

## 二、进阶使用

### 1. 采用Flash Attention 2.0

为了能够使用其框架进行加速，首先需要在对应的环境中安装对应的依赖。  

```bash
conda activate siglip
pip install flash-attn==2.7.3 accelerate==1.3.0
```

安装完成上述依赖后，使用Flash Attention 2.0针对注意力头进行加速推理。

```python
import torch
import requests
from PIL import Image
from transformers import SiglipProcessor, SiglipModel
device = "cuda"

model = SiglipModel.from_pretrained(
    "/root/autodl-tmp/siglip-so400m-patch14-384",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,
    device_map=device,
)
processor = SiglipProcessor.from_pretrained("/root/autodl-tmp/siglip-so400m-patch14-384")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

candidate_labels = ["2 cats", "2 dogs"]
texts = [f'This is a photo of {label}.' for label in candidate_labels]
inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")
inputs.to(device)

with torch.no_grad():
    with torch.autocast(device, dtype=torch.float16):
        outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image)
print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")
```

### 2. 使用缩放点积注意力 （SDPA）

PyTorch包含一个原生缩放点积注意力 （SDPA） 运算符，作为 torch.nn.functional 的一部分。此功能 包含多种实现，这些实现可以根据 inputs 和使用的硬件进行应用，这里需要torch>=2.1.1才可以使用。

```python
# 基于上述例子只要修改其中部分即可
model = SiglipModel.from_pretrained(
    "/root/autodl-tmp/siglip-so400m-patch14-384",
    attn_implementation="sdpa",
    torch_dtype=torch.float16,
    device_map=device,
)
```

关于上述基于Flash Attention 2.0 与 SDPA 的的性能对比差异如下图所示

![](/media/17380499894001.png)

## 三、微调模型

针对多模态的使用场景中，针对现实世界中以及特定领域场景的使用往往需要补充针对性的图片-文字对实现更高效的推理能力，为此
下述将基于上述目前最优的模型基础上进行微调以实现特定场景数据的适配。由于推理对于机器的要求不高，但是在进行数据训练则需要
满足一定要求的机器配置才可以顺利的进行微调推理，为此需要满足以下最低配置的要求。

微调系统环境
* CPU: 6 vCPU Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz
* MEM: 25GB
* GPU: V100-32GB(32GB)
* 系统: ubuntu22.04
* 环境: PyTorch  2.3.0 + Python 3.12 +Cuda  12.1

### 1. 测试数据

继上述已创建的siglip环境，还需要安装以下需要的库。

```bash
# 安装额外的库
pip install -q datasets

# 下载本次使用的数据集
curl -L -o ~/autodl-tmp/multi-label-image-classification-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/meherunnesashraboni/multi-label-image-classification-dataset
unzip multi-label-image-classification-dataset.zip
```

### 2. 快速训练

将下载的csv读取为Pandas数据集，每行都包含一个训练示例，其中包含图像的文件名和相应的 one-hot 编码标签。
```python
import pandas as pd

df = pd.read_csv("/root/autodl-tmp/multilabel_modified/multilabel_classification(2).csv")
df.head()
```

创建一个 id2label 字典，将整数映射到字符串。
```python
labels = list(df.columns)[2:]
id2label = {id: label for id, label in enumerate(labels)}
print(id2label)
```

接下来加载离线模型与图像处理器，其中将problem_type指定为 “multi_label_classification”,其是告诉模型当前为多标签分类，从而促使其使用正确的激活函数，
```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "/root/autodl-tmp/siglip-so400m-patch14-384" # 指定为实际本地离线模型路径

processor = AutoImageProcessor.from_pretrained(model_id, device=device)
model = AutoModelForImageClassification.from_pretrained(model_id, problem_type="multi_label_classification", id2label=id2label)
model = model.to(device) 
```

创建数据集读取类，从而确保能够正确的读取图片以及分类标签并转换为正确的格式
```python
from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import numpy as np

class MultiLabelDataset(Dataset):
  def __init__(self, root, df, transform):
    self.root = root
    self.df = df
    self.transform = transform

  def __getitem__(self, idx):
    item = self.df.iloc[idx]
    # get image
    image_path = os.path.join(self.root, item["Image_Name"])

    if not os.path.exists(image_path):
        return None

    image = Image.open(image_path).convert("RGB")

    # prepare image for the model
    pixel_values = self.transform(image)

    # get labels
    labels = item[2:].values.astype(np.float32)

    # turn into PyTorch tensor
    labels = torch.from_numpy(labels)

    return pixel_values, labels

  def __len__(self):
    return len(self.df)
```

为了准备的图像，将使用 Torchvision 包，它提供了若干图像转换工具将图像大小调整为模型预期的大小（在本例中为 384），
并且使用适当的平均值和标准偏差对颜色通道进行标准化。
```python
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# get appropriate size, mean and std based on the image processor
size = processor.size["height"]
mean = processor.image_mean
std = processor.image_std

transform = Compose([
    Resize((size, size)),
    ToTensor(),
    Normalize(mean=mean, std=std),
])

train_dataset = MultiLabelDataset(root="/root/autodl-tmp/multilabel_modified/images",
                                  df=df, transform=transform)
len(train_dataset)
```

接下来，我们可以创建相应的 PyTorch DataLoader，以获取批量训练示例（因为神经网络通常使用随机梯度下降 = SGD 对批量数据进行训练）。
```python
from torch.utils.data import DataLoader

def collate_fn(batch):
    # 过滤掉 None
    batch = [item for item in batch if item is not None]
    
    # 如果 batch 为空，返回 None，避免 torch.stack 出错
    if len(batch) == 0:
        return None

    data = torch.stack([item[0] for item in batch])
    target = torch.stack([item[1] for item in batch])
    return data, target

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=2, shuffle=True)
batch = next(iter(train_dataloader))

# 验证初始损失
outputs = model(pixel_values=batch[0].to(device), labels=batch[1].to(device))
outputs.loss
```

是时候训练模型了！我们将在此处以常规的 PyTorch 方式进行训练，但请随时升级以利用 🤗 Accelerate（对于具有最少代码更改的分布式训练非常有用），或者利用 🤗 Trainer 类来处理
我们在此处为您定义的许多逻辑（例如创建数据加载器）。
- learning rate  学习率
- number of epochs  纪元数
- optimizer  优化
- gradient accumulation, gradient checkpointing, Flash Attention can be leveraged to speed up training 可以利用梯度累积、梯度检查点、Flash Attention 来加速训练
- mixed precision training (bfloat16) etc. 混合精度训练 （bfloat16） 等。
```python
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

from torch.optim import AdamW
from tqdm.auto import tqdm

optimizer = AdamW(model.parameters(), lr=5e-5)

losses = AverageMeter()

model.train()
for epoch in range(10):  # loop over the dataset multiple times
    for idx, batch in enumerate(tqdm(train_dataloader)):
        # 跳过无效批次
        if batch is None:
            continue
        # get the inputs;
        pixel_values, labels = batch

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(
            pixel_values=pixel_values.to(device),
            labels=labels.to(device),
        )

        # calculate gradients
        loss = outputs.loss
        losses.update(loss.item(), pixel_values.size(0))
        loss.backward()

        # optimization step
        optimizer.step()

        if idx % 2000 == 0:
            print('Epoch: [{0}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, loss=losses,))
```

测试进行推理
```python
image = Image.open("/root/autodl-tmp/multilabel_modified/images/image6179.jpg")
model.eval()

# prepare image for the model
pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

# forward pass
with torch.no_grad():
  outputs = model(pixel_values)
  logits = outputs.logits

# 由于我们在训练期间使用了 BCEWithLogitsLoss（在计算损失之前对 logit 应用 sigmoid），因此我们也需要在此处将 sigmoid 应用于 logits。这将它们转化为单独的概率。
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(logits.squeeze().cpu())

# select the probabilities > a certain threshold (e.g. 50%) as predicted
predictions = np.zeros(probs.shape)
predictions[np.where(probs >= 0.5)] = 1 # turn predicted id's into actual label names
predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
print(predicted_labels)
```

持久化保存模型
```python
model.save_pretrained("./saved_model/")
```

## 四、特征提取

本仓库中新增了 `siglip_vision_extractor.py` 文件，该脚本用于独立提取图片/视频帧的视觉特征，
并进一步展示如何将这些特征传入其他大语言模型（例如基于 langchain 实现的 qwen 或 deepseek）。

### 功能概述

- **视觉特征提取**: 利用预训练的 Siglip 模型对图像进行特征提取，其输出张量形状为 (B, T, W, H, hidden_size)。
- **特征图可视化**: 脚本中使用 matplotlib 将特征隐藏维度的平均值展示为热力图。
- **与 LLM 结合**: 提供示例函数 `send_features_to_llm` 展示如何将提取的视觉特征转换为提示信息传入大语言模型。

### 使用方法

1. **依赖安装**: 确保安装以下依赖：
   - torch
   - transformers
   - pillow (PIL)
   - opencv-python
   - matplotlib (可选，用于可视化)

2. **集成使用**:
   `siglip_vision_extractor.py` 文件并非设计为独立运行的脚本，而是提供了一个用于提取视觉特征的类 `SiglipVisionTower`，
   供用户在多模态应用中调用。你可以通过直接导入模块并实例化该类来集成使用：

   ```python
   from siglip.siglip_vision_extractor import SiglipVisionTower, VisionTowerConfig

   # 设置模型路径与配置参数（请替换为实际模型名称或路径）
   model_name_or_path = "siglip-base"
   config = VisionTowerConfig(
       vision_tower_name=model_name_or_path,
       img_size=224,       
       patch_size=16,      
       hidden_size=768,    
       num_frames=1        
   )

   # 初始化 SiglipVisionTower 模型
   model = SiglipVisionTower(model_name_or_path, config)

   # 加载图像并提取视觉特征
   from PIL import Image
   img = Image.open("example.jpg").convert("RGB")  # 请确保存在示例图像
   inputs = model.vision_processor(img, return_tensors="pt")
   features = model(inputs["pixel_values"])
   print("Extracted features shape:", features.shape)
   ```

   你可以在此基础上进一步设计降维或线性映射模块，将提取的高维视觉特征转换成适合大语言模型的输入格式，并结合其他 LLM（例如基于 langchain 的 qwen 或 deepseek）使用。
