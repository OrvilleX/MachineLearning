# InternVideo 多模态视频理解模型

通过动态筛选关键帧降低计算成本，结合视觉、文本、音频实现高效跨模态分析。核心解决视频时空复杂度高、多模态融合难、标注依赖强的问题，适用于行为识别、视频检索、问答等场景，优势为高效、灵活、开源，支持轻量化部署与复杂任务适配。

参考文档
* [官方代码仓库](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2/multi_modality)  
* [HF模型仓库集](https://huggingface.co/collections/OpenGVLab/internvideo2-6618ccb574bd2f91410df5cd)  
* [分词模型bert-large-uncased](https://huggingface.co/google-bert/bert-large-uncased)

推理系统环境
* CPU: 18 vCPU AMD EPYC 9754 128-Core Processor
* MEM: 60GB
* GPU: RTX 3090(24GB) * 1
* 系统: ubuntu22.04
* 环境: PyTorch  2.5.1 + Python 3.12 + Cuda  12.1

## 一、快速使用

### 1. 系统配置

本次推理主要使用的为[AutoDL](https://www.autodl.com/)云算力租用平台，部分安装指令是针对其主机生效的，将
会单独标记出来表示。

```bash
# AutoDL中将conda安装环境调整到数据盘
mkdir -p /root/autodl-tmp/conda/pkgs
conda config --add pkgs_dirs /root/autodl-tmp/conda/pkgs

mkdir -p /root/autodl-tmp/conda/envs
conda config --add envs_dirs /root/autodl-tmp/conda/envs

# 安装需要的系统库
sudo apt update
sudo apt install nload git-lfs libaio-dev -y
git lfs install
```

### 2. 环境配置

考虑到实际环境的隔离性，为此这里使用conda环境独立的虚拟环境，并安装对应的依赖库。

```bash
# 创建独立的conda环境
conda create -y -n iv python=3.10 -q
conda init bash && source /root/.bashrc

# 安装依赖库
conda activate iv
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

配置到jupyterlab
conda install ipykernel
ipython kernel install --user --name=iv
pip install jupyterlab_widgets
```

### 3. 项目配置

由于项目需要依赖使用其他额外的库，所以下面首先针对这些额外的库进行安装编译并配置。首先是安装
对应的flash-attn以及对应的插件（FusedMLP and DropoutLayerNorm）。

```bash
conda activate iv
pip install flash-attn==2.6.3 --no-build-isolation

# 下载flash-attn源码并编译插件
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention

# 切换到符合的版本
git checkout v2.6.3

# 安装fused_mlp_lib
cd csrc/fused_dense_lib && pip install .

# 安装layer_form，其中MAX_JOBS根据目前自己的内存调整，否则编译会出现因为OOM导致的killed
export MAX_JOBS=18
cd csrc/layer_norm && pip install .
```

接着还需要通过源码安装open-clip库

```bash
pip install git+https://github.com/mlfoundations/open_clip.git
```

最后就是安装项目并配置下载所需的模型文件，下述下载的模型与源码均在一个层级文件夹下。

```bash
# 下载项目源码
git clone https://github.com/OpenGVLab/InternVideo.git
cd InternVideo/InternVideo2/multi_modality

# 安装项目依赖
pip install -r requirements.txt

# 下载分词模型
git clone https://huggingface.co/google-bert/bert-large-uncased

# 下载InternVideo2模型
git clone https://huggingface.co/OpenGVLab/InternVideo2-Stage2-1B-224p-f4
```

完成分词模型的下载后还需要修改项目源码中的配置，打开`multi_modality/demo/utils.py`文件定位到`setup_internvideo2`函数
将下述代码片段中的`your_model_path`修改为对应模型的文件夹即可。

```python
tokenizer = BertTokenizer.from_pretrained('your_model_path', local_files_only=True)
```

> 同时还有对应的`config/model.py`中对应的`bert`中的配置文件`config`以及模型的路径`pretrained`也需要调整

```python
TextEncoders["bert"] = dict(
    name="bert_base",
    pretrained="bert-base-uncased",
    config="configs/config_bert_large.json",
    d_model=768,
    fusion_layer=9,
)
```

接着就是配置InternVideo2模型路径，打开`multi_modality/demo/internvideo2_stage2_config.py`文件，将其中的`your_model_path`替换
为具体的模型文件路径（pt模型结尾的文件）。

```python
model = dict(
    model_cls="InternVideo2_Stage2",
    vision_encoder=dict(
        ...
        pretrained='your_model_path',
        ...
    )
)
```

### 4. 运行demo

接着打开`demo_video_text_retrieval.ipynb`运行其中的例子，正确运行后会输出如下的结果。  

```plaintext
text: A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run. ~ prob: 0.7927
text: A playful dog and its owner wrestle in the snowy yard, chasing each other with joyous abandon. ~ prob: 0.1769
text: A pet dog excitedly runs through the snowy yard, chasing a toy thrown by its owner. ~ prob: 0.0291
text: A man in a gray hat and coat walks through the snowy yard, carefully navigating around the trees. ~ prob: 0.0006
text: A person dressed in a blue jacket shovels the snow-covered pavement outside their house. ~ prob: 0.0003
```