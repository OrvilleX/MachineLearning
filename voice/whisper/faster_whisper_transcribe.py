import os
import sys
import tempfile
import traceback
import logging
from typing import Dict, Generator

# 第三方库导入
import numpy as np
import torch
from faster_whisper import WhisperModel
import librosa
import soundfile as sf

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FasterWhisperTranscribe")


class FasterWhisperTranscribe:
    def __init__(self, 
                 whisper_model: str = "large-v3", 
                 device: str = "cuda",
                 compute_type: str = "int8",
                 chunk_size: int = 20,
                 max_file_size: int = 1 * 1024 * 1024,
                 vad_threshold: float = 0.5,
                 beam_size: int = 5,
                 best_of: int = 5,
                 temperature: float = 0.0,
                 compression_ratio_threshold: float = 2.4,
                 log_prob_threshold: float = -1.0,
                 no_speech_threshold: float = 0.6,
                 condition_on_previous_text: bool = True,
                 initial_prompt: str = None,
                 word_timestamps: bool = False,
                 debug: bool = False):
        """
        初始化音频转换服务
        
        Args:
            whisper_model: whisper模型路径或预训练模型名称
                           例如: "large-v3", "medium", "small", "base", "tiny"
            device: 计算设备，默认为"cuda"
                   可选值: "cuda", "cpu"
            compute_type: 计算类型，默认为"int8"
                         可选值: "float16", "float32", "int8"
            chunk_size: 大文件分块大小（秒），默认为30秒
            max_file_size: 触发分块处理的文件大小阈值（字节），默认为1MB
            vad_threshold: VAD阈值，控制语音检测的敏感度，默认为0.5
            beam_size: 束搜索大小，值越大准确度越高但速度越慢，默认为5
            best_of: 采样数量，值越大准确度越高但速度越慢，默认为5
            temperature: 采样温度，值越高多样性越大但准确度可能降低，默认为0.0
            compression_ratio_threshold: 压缩比阈值，用于过滤重复内容，默认为2.4
            log_prob_threshold: 对数概率阈值，用于过滤低概率内容，默认为-1.0
            no_speech_threshold: 无语音阈值，控制静音检测的敏感度，默认为0.6
            condition_on_previous_text: 是否基于前文进行预测，提高连贯性，默认为True
            initial_prompt: 初始提示文本，可以提高特定领域的准确度，默认为None
            word_timestamps: 是否生成单词级时间戳，默认为False
            debug: 是否启用调试模式，默认为False
        """
        # 设置调试模式
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        # 初始化Whisper模型
        compute_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        logger.debug(f"使用设备: {compute_device}, 计算类型: {compute_type}")
        
        try:
            self.model = WhisperModel(whisper_model, device=compute_device, compute_type=compute_type)
            logger.debug(f"成功加载Whisper模型: {whisper_model}")
        except Exception as e:
            logger.error(f"加载Whisper模型失败: {str(e)}")
            raise RuntimeError(f"无法加载Whisper模型: {str(e)}")
        
        # 配置参数
        self.chunk_size = chunk_size
        self.max_file_size = max_file_size
        self.vad_threshold = vad_threshold
        
        # 转写参数
        self.beam_size = beam_size
        self.best_of = best_of
        self.temperature = temperature
        self.compression_ratio_threshold = compression_ratio_threshold
        self.log_prob_threshold = log_prob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.condition_on_previous_text = condition_on_previous_text
        self.initial_prompt = initial_prompt
        self.word_timestamps = word_timestamps
        
        logger.debug(f"初始化参数 - 分块大小: {chunk_size}秒, 最大文件大小: {max_file_size/1024/1024}MB, VAD阈值: {vad_threshold}")
        logger.debug(f"转写参数 - 束搜索大小: {beam_size}, 采样数量: {best_of}, 温度: {temperature}")

    def transcribe_file(self, audio_path: str, language: str = "zh") -> Generator[Dict, None, None]:
        """
        转写音频文件为文本，返回迭代器
        
        对于大型文件，会自动进行分块处理
        
        Args:
            audio_path: 音频文件路径
            language: 语言代码，默认为中文"zh"
                     其他常用值: "en"(英语), "ja"(日语), "ko"(韩语)等
            
        Returns:
            生成器，每次产生一个包含转写结果的字典
            
        示例:
        ```python
        for segment in transcriber.transcribe_file("path/to/audio.wav"):
            print(f"{segment['start']}s - {segment['end']}s: {segment['text']}")
        ```
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(audio_path):
                logger.error(f"文件不存在: {audio_path}")
                raise FileNotFoundError(f"文件不存在: {audio_path}")
            
            # 检查文件大小
            file_size = os.path.getsize(audio_path)
            logger.debug(f"文件大小: {file_size/1024/1024:.2f}MB")
            
            # 如果文件较小，直接处理
            if file_size <= self.max_file_size:
                logger.debug(f"文件大小小于阈值，直接处理")
                yield from self._transcribe_audio_direct(audio_path, language)
            else:
                # 对大文件进行分块处理
                logger.debug(f"文件大小超过阈值，进行分块处理")
                yield from self._transcribe_audio_chunked(audio_path, language)
            
        except Exception as e:
            logger.error(f"音频转写异常: {str(e)}")
            if self.debug:
                logger.error(traceback.format_exc())
            raise

    def transcribe_file_direct(self, audio_path: str, language: str = "zh") -> Generator[Dict, None, None]:
        """
        直接转写音频文件为文本，不进行分块处理
        
        Args:
            audio_path: 音频文件路径
            language: 语言代码，默认为中文"zh"
            
        Returns:
            生成器，每次产生一个包含转写结果的字典
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(audio_path):
                logger.error(f"文件不存在: {audio_path}")
                raise FileNotFoundError(f"文件不存在: {audio_path}")
            
            yield from self._transcribe_audio_direct(audio_path, language)
            
        except Exception as e:
            logger.error(f"音频直接转写异常: {str(e)}")
            if self.debug:
                logger.error(traceback.format_exc())
            raise

    def transcribe_file_chunked(self, audio_path: str, language: str = "zh") -> Generator[Dict, None, None]:
        """
        分块转写音频文件为文本
        
        Args:
            audio_path: 音频文件路径
            language: 语言代码，默认为中文"zh"
            
        Returns:
            生成器，每次产生一个包含转写结果的字典
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(audio_path):
                logger.error(f"文件不存在: {audio_path}")
                raise FileNotFoundError(f"文件不存在: {audio_path}")
            
            yield from self._transcribe_audio_chunked(audio_path, language)
            
        except Exception as e:
            logger.error(f"音频分块转写异常: {str(e)}")
            if self.debug:
                logger.error(traceback.format_exc())
            raise

    def _transcribe_audio_direct(self, audio_path: str, language: str) -> Generator[Dict, None, None]:
        """直接处理音频文件（不分块）"""
        logger.debug(f"开始直接转写: {audio_path}")
        
        try:
            segments, info = self.model.transcribe(
                audio_path, 
                language=language,
                beam_size=self.beam_size,
                best_of=self.best_of,
                temperature=self.temperature,
                compression_ratio_threshold=self.compression_ratio_threshold,
                log_prob_threshold=self.log_prob_threshold,
                no_speech_threshold=self.no_speech_threshold,
                condition_on_previous_text=self.condition_on_previous_text,
                initial_prompt=self.initial_prompt,
                word_timestamps=self.word_timestamps,
                without_timestamps=False
            )
            
            detected_language = info.language if hasattr(info, 'language') else language
            logger.debug(f"检测到语言: {detected_language}")
            
            # 处理结果
            for segment in segments:
                segment_dict = {
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "language": detected_language
                }
                
                # 如果启用了单词级时间戳，添加单词信息
                if self.word_timestamps and hasattr(segment, 'words') and segment.words:
                    segment_dict["words"] = [
                        {"word": word.word, "start": word.start, "end": word.end, "probability": word.probability}
                        for word in segment.words
                    ]
                
                logger.debug(f"转写片段: {segment.start:.2f}s - {segment.end:.2f}s")
                yield segment_dict
                
        except Exception as e:
            logger.error(f"直接转写处理异常: {str(e)}")
            if self.debug:
                logger.error(traceback.format_exc())
            raise

    def _transcribe_audio_chunked(self, audio_path: str, language: str) -> Generator[Dict, None, None]:
        """分块处理大型音频文件"""
        logger.debug(f"开始分块转写: {audio_path}")
        
        try:
            # 加载音频文件信息
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            
            logger.debug(f"音频时长: {duration:.2f}秒, 采样率: {sr}Hz")
            logger.debug(f"分块大小: {self.chunk_size}秒")
            
            detected_language = None
            
            # 分块处理
            for start_time in range(0, int(duration), self.chunk_size):
                end_time = min(start_time + self.chunk_size, duration)
                logger.debug(f"处理时间段: {start_time}s - {end_time}s")
                
                # 提取音频片段
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                chunk = y[start_sample:end_sample]
                
                # 保存临时文件
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_path = temp_file.name
                    sf.write(temp_path, chunk, sr)
                
                try:
                    # 处理该片段
                    segments, info = self.model.transcribe(
                        temp_path,
                        language=language,
                        beam_size=self.beam_size,
                        best_of=self.best_of,
                        temperature=self.temperature,
                        compression_ratio_threshold=self.compression_ratio_threshold,
                        log_prob_threshold=self.log_prob_threshold,
                        no_speech_threshold=self.no_speech_threshold,
                        condition_on_previous_text=self.condition_on_previous_text,
                        initial_prompt=self.initial_prompt,
                        word_timestamps=self.word_timestamps,
                        without_timestamps=False
                    )
                    
                    # 记录检测到的语言（使用第一个块的语言）
                    if detected_language is None and hasattr(info, 'language'):
                        detected_language = info.language
                        logger.debug(f"检测到语言: {detected_language}")
                    
                    # 处理结果
                    for segment in segments:
                        segment_dict = {
                            "text": segment.text,
                            "start": segment.start + start_time,  # 调整时间戳
                            "end": segment.end + start_time,      # 调整时间戳
                            "language": detected_language or language
                        }
                        
                        # 如果启用了单词级时间戳，添加单词信息并调整时间戳
                        if self.word_timestamps and hasattr(segment, 'words') and segment.words:
                            segment_dict["words"] = [
                                {
                                    "word": word.word, 
                                    "start": word.start + start_time, 
                                    "end": word.end + start_time, 
                                    "probability": word.probability
                                }
                                for word in segment.words
                            ]
                        
                        logger.debug(f"转写片段: {segment_dict['start']:.2f}s - {segment_dict['end']:.2f}s")
                        yield segment_dict
                
                finally:
                    # 删除临时文件
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        logger.warning(f"删除临时文件失败: {str(e)}")
            
        except Exception as e:
            logger.error(f"分块处理异常: {str(e)}")
            if self.debug:
                logger.error(traceback.format_exc())
            raise


# 使用示例
if __name__ == "__main__":
    # 手动设置参数，便于调整
    audio_path = "/root/autodl-tmp/longaudio.wav"  # 输入音频文件路径
    whisper_model = "/root/autodl-tmp/faster-whisper-large-v3-zh"  # whisper模型
    output_path = "transcription_result.txt"  # 输出文本文件路径
    language = "zh"  # 语言代码
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 计算设备
    debug = True  # 是否启用调试模式
    
    try:
        # 初始化转写器，可以根据需要调整参数以优化准确度
        transcriber = FasterWhisperTranscribe(
            whisper_model=whisper_model,
            device=device,
            beam_size=5,           # 增大可提高准确度，但会降低速度
            best_of=5,             # 增大可提高准确度，但会降低速度
            temperature=0.0,       # 0表示贪婪解码，提高准确度
            no_speech_threshold=0.6,  # 调整静音检测敏感度
            initial_prompt=None,   # 可以添加领域相关提示以提高准确度
            word_timestamps=False, # 是否需要单词级时间戳
            debug=debug
        )
        
        print(f"处理音频文件: {audio_path}")
        
        # 收集所有转写结果
        all_segments = []
        full_text = ""
        
        # 使用迭代器处理转写结果
        for segment in transcriber.transcribe_file(audio_path, language):
            all_segments.append(segment)
            full_text += segment["text"] + " "
            
            # 实时输出转写结果
            print(f"{segment['start']:.2f}s - {segment['end']:.2f}s: {segment['text']}")
        
        # 输出完整结果
        print("\n完整转写结果:")
        print(full_text.strip())
        
        # 保存结果到文件
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_text.strip())
            print(f"\n转写结果已保存到: {output_path}")
            
    except Exception as e:
        print(f"处理异常: {str(e)}")
        if debug:
            traceback.print_exc()
        sys.exit(1)