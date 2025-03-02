# whisper

参考文档
* [hf国内调优后模型](https://huggingface.co/BELLE-2/Belle-whisper-large-v3-turbo-zh)
* [微调文档](https://github.com/shuaijiang/Whisper-Finetune)  

## 一、快速使用

### 1.1 系统配置

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
sudo apt install nload git-lfs ffmpeg -y

git lfs install
```

### 1.2 环境配置

考虑到实际环境的隔离性，为此这里使用conda环境独立的虚拟环境，并安装对应的依赖库。

```bash
# 创建独立的conda环境
conda create -y -n whisper python=3.10 -q
conda init bash && source /root/.bashrc

# 安装依赖库
conda activate whisper
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

配置到jupyterlab
conda install ipykernel
ipython kernel install --user --name=whisper
pip install jupyterlab_widgets
```

### 1.3 项目配置

```bash
# 下载官方模型
git clone https://huggingface.co/BELLE-2/Belle-whisper-large-v3-turbo-zh

# 国内加速
git clone https://hf-mirror.com/BELLE-2/Belle-whisper-large-v3-turbo-zh

# 安装必要依赖
pip install transformers==4.37.2 ffmpeg-python faster-whisper==0.10.1 librosa soundfile numpy
```

### 1.4 运行demo

```python
from transformers import pipeline

transcriber = pipeline(
  "automatic-speech-recognition", 
  model="BELLE-2/Belle-whisper-large-v3-turbo-zh"
)

transcriber.model.config.forced_decoder_ids = (
  transcriber.tokenizer.get_decoder_prompt_ids(
    language="zh", 
    task="transcribe"
  )
)

transcription = transcriber("my_audio.wav")
```

## 二、实时音频流处理

### 2.1 实时音频流模拟

为了能够模拟实时的音频流的处理为此需要使用组件将文件模拟为对应的流从而能够实现此类功能，为此这里
使用`icecast2`与`liquidsoap`组件来实现。  

```bash
# 安装流媒体服务
sudo apt update
sudo apt install icecast2

# 修改配置（如端口等）
vi /etc/icecast2/icecast.xml

# 启动流媒体服务
sudo service icecast2 start

# 安装流推送工具
sudo apt install liquidsoap

# 修改推送配置
vi stream.liq

    set("init.allow_root", true)
    
    # Load the WAV file
    audio = single("/root/autodl-tmp/output.wav")
    
    # Define the Icecast output configuration
    output.icecast(%mp3, host="localhost", port=6006, mount="/stream", password="hackme", audio)

# 启动服务
liquidsoap stream.liq
```

如果是需要将mp4文件中的音频进行推送可以采用`ffmpeg -stream_loop -1 -i /root/autodl-tmp/video.mp4 -vn -acodec libmp3lame -ab 128k -f mp3 icecast://source:hackme@localhost:6006/stream`进行推送

### 2.2 进行音转文

由于faster-whisper本身并不具备针对实时音频流的处理，所以这里需要先将音频根据要求进行流读取并切分为对应的本地片段，紧接着就是让`faster-whisper`进行处理，为此需要通过[仓库](./real_time_transcribe.ipynb)中代码进行运行从而实现针对音频流的读取与分析。

## 三、核心类说明

### 3.1 FasterWhisperTranscribe

`FasterWhisperTranscribe` 是对 faster-whisper 的封装，提供了更便捷的音频转写功能。

#### 初始化参数

```python
FasterWhisperTranscribe(
    whisper_model: str = "large-v3",      # 模型路径或预训练模型名称
    device: str = "cuda",                 # 计算设备，"cuda"或"cpu"
    compute_type: str = "int8",           # 计算类型，"float16"、"float32"或"int8"
    chunk_size: int = 20,                 # 大文件分块大小（秒）
    max_file_size: int = 1 * 1024 * 1024, # 触发分块处理的文件大小阈值（字节）
    vad_threshold: float = 0.5,           # VAD阈值，控制语音检测敏感度
    beam_size: int = 5,                   # 束搜索大小，值越大准确度越高但速度越慢
    best_of: int = 5,                     # 采样数量，值越大准确度越高但速度越慢
    temperature: float = 0.0,             # 采样温度，值越高多样性越大但准确度可能降低
    compression_ratio_threshold: float = 2.4,  # 压缩比阈值，用于过滤重复内容
    log_prob_threshold: float = -1.0,     # 对数概率阈值，用于过滤低概率内容
    no_speech_threshold: float = 0.6,     # 无语音阈值，控制静音检测敏感度
    condition_on_previous_text: bool = True,  # 是否基于前文进行预测，提高连贯性
    initial_prompt: str = None,           # 初始提示文本，可提高特定领域准确度
    word_timestamps: bool = False,        # 是否生成单词级时间戳
    debug: bool = False                   # 是否启用调试模式
)
```

#### 主要方法

1. **transcribe_file**

   转写音频文件为文本，返回迭代器。对于大型文件，会自动进行分块处理。

   ```python
   transcribe_file(audio_path: str, language: str = "zh") -> Generator[Dict, None, None]
   ```

   参数:
   - `audio_path`: 音频文件路径
   - `language`: 语言代码，默认为中文"zh"

   返回:
   - 生成器，每次产生一个包含转写结果的字典

   示例:
   ```python
   for segment in transcriber.transcribe_file("path/to/audio.wav"):
       print(f"{segment['start']}s - {segment['end']}s: {segment['text']}")
   ```

2. **transcribe_file_direct**

   直接转写音频文件为文本，不进行分块处理。

   ```python
   transcribe_file_direct(audio_path: str, language: str = "zh") -> Generator[Dict, None, None]
   ```

   参数与返回值同 `transcribe_file`。

3. **transcribe_file_chunked**

   分块转写音频文件为文本。

   ```python
   transcribe_file_chunked(audio_path: str, language: str = "zh") -> Generator[Dict, None, None]
   ```

   参数与返回值同 `transcribe_file`。

#### 返回结果格式

每个转写段的结果是一个字典，包含以下字段:

```python
{
    "text": "转写的文本内容",
    "start": 0.0,  # 开始时间（秒）
    "end": 5.2,    # 结束时间（秒）
    "language": "zh"  # 检测到的语言
}
```

如果启用了单词级时间戳 (`word_timestamps=True`)，结果还会包含 `words` 字段:

```python
{
    # ... 其他字段 ...
    "words": [
        {"word": "你好", "start": 0.0, "end": 0.5, "probability": 0.98},
        {"word": "世界", "start": 0.6, "end": 1.2, "probability": 0.99},
        # ... 更多单词 ...
    ]
}
```

### 3.2 AudioStreamProcessor

`AudioStreamProcessor` 用于处理实时音频流，将其分割成固定长度的音频段。

#### 初始化参数

```python
AudioStreamProcessor(
    stream_url: str,                    # 音频流URL (支持http/https/rtsp/rtmp)
    segment_duration: int = 10,         # 每个音频段的持续时间（秒）
    sample_rate: int = 16000,           # 采样率（Hz）
    channels: int = 1,                  # 音频通道数
    output_format: str = "wav",         # 输出格式 (wav/flac)
    debug: bool = False,                # 是否启用调试模式
    debug_dir: str = "./debug_audio",   # 调试模式下保存音频文件的目录
    log_level: int = logging.INFO,      # 日志级别
    max_retries: int = 3,               # 连接失败时的最大重试次数
    retry_delay: int = 5,               # 重试之间的延迟（秒）
    ffmpeg_loglevel: str = "warning"    # ffmpeg日志级别
)
```

#### 主要方法

1. **start**

   开始处理音频流并返回音频段文件路径的迭代器。

   ```python
   start() -> Iterator[str]
   ```

   返回:
   - 迭代器，每次产生一个音频段文件的路径

   示例:
   ```python
   processor = AudioStreamProcessor(stream_url="http://example.com/audio_stream")
   for segment_file in processor.start():
       print(f"生成音频段: {segment_file}")
       # 处理音频段...
   ```

2. **pause**

   暂停处理音频流。

   ```python
   pause() -> None
   ```

3. **resume**

   恢复处理音频流。

   ```python
   resume() -> None
   ```

4. **stop**

   停止处理音频流。

   ```python
   stop() -> None
   ```

5. **cleanup**

   清理临时文件。

   ```python
   cleanup() -> None
   ```
