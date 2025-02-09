# Kokoro TTS技术

Kokoro TTS 是一款轻量级、开源的文本转语音模型，其参数仅 8200 万，却能生成高质量、自然流畅的语音。该模型基于 StyleTTS2 与 ISTFTNet 架构，通过精心挑选的非版权音频数据训练，实现高效实时推理。支持美式、英式英语、法语、日语、韩语及中文等多语言，适用于有声书、播客、培训视频及数字无障碍服务。Kokoro TTS 可在本地 CPU 或 GPU 环境下运行。

参考文档
* [Kokoro-82M模型hf仓库](https://huggingface.co/hexgrad/Kokoro-82M)  
* [kokoro开源代码仓库](https://github.com/hexgrad/kokoro)  

其他高级使用方式


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
sudo apt install nload git-lfs -y
sudo apt-get -qq -y install espeak-ng > /dev/null 2>&1

git lfs install
```

### 2. 环境配置

考虑到实际环境的隔离性，为此这里使用conda环境独立的虚拟环境，并安装对应的依赖库。

```bash
# 创建独立的conda环境
conda create -y -n kk python=3.10 -q
conda init bash && source /root/.bashrc

# 安装依赖库
conda activate kk
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

配置到jupyterlab
conda install ipykernel
ipython kernel install --user --name=kk
pip install jupyterlab_widgets
```

### 3. 项目配置

```bash
# 下载官方模型
git clone https://huggingface.co/hexgrad/Kokoro-82M

# 国内加速
git clone https://hf-mirror.com/hexgrad/Kokoro-82M


conda activate kk
pip install -q kokoro>=0.3.4 soundfile misaki[zh]
```

### 4. 运行demo

接着可以采用下面的代码运行对应的demo将指定的中文转换为所需的音频文件。

```python
from kokoro import KPipeline, KModel
from IPython.display import display, Audio
import soundfile as sf

# 参数中config与model修改为本地模型路径
model = KModel(config='/root/autodl-tmp/Kokoro-82M/config.json', model='/root/autodl-tmp/Kokoro-82M/kokoro-v1_0.pth')
pipeline = KPipeline(lang_code='z', model=model)
text = '中國人民不信邪也不怕邪，不惹事也不怕事，任何外國不要指望我們會拿自己的核心利益做交易，不要指望我們會吞下損害我國主權、安全、發展利益的苦果！'

# 下方voice修改为本地对应模型路径
generator = pipeline(
    text, voice='/root/autodl-tmp/Kokoro-82M/voices/zm_yunxi.pt', # <= change voice here
    speed=0.85, split_pattern=r'\n+'
)
for i, (gs, ps, audio) in enumerate(generator):
    print(i)  # i => index
    print(gs) # gs => graphemes/text
    print(ps) # ps => phonemes
    display(Audio(data=audio, rate=24000, autoplay=i==0))
    sf.write(f'{i}.wav', audio, 24000)
```
