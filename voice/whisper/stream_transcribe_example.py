import time
import threading
import queue
import logging
import traceback
from typing import Dict, List

import torch

# 导入自定义模块
from audio_stream_processor import AudioStreamProcessor
from faster_whisper_transcribe import FasterWhisperTranscribe

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StreamTranscribe")


def main():
    """主函数 - 直接使用AudioStreamProcessor和FasterWhisperTranscribe实现实时音频流转写"""
    
    # 直接定义参数变量
    # 音频流参数
    stream_url = "http://localhost:6006/stream"  # 音频流URL (支持http/https/rtsp/rtmp)
    segment_duration = 10  # 每个音频段的持续时间（秒）
    sample_rate = 16000  # 采样率（Hz）
    channels = 1  # 音频通道数
    output_format = "wav"  # 输出格式 (wav/flac)
    
    # Whisper模型参数
    whisper_model = "/root/autodl-tmp/faster-whisper-large-v3-zh"  # whisper模型路径或预训练模型名称
    language = "zh"  # 转写语言代码
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 计算设备
    compute_type = "int8"  # 计算类型
    
    # 其他参数
    debug = False  # 启用调试模式
    max_queue_size = 10  # 最大队列大小
    
    # 设置日志级别
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    try:
        # 初始化音频流处理器
        logger.info(f"初始化音频流处理器: {stream_url}")
        audio_processor = AudioStreamProcessor(
            stream_url=stream_url,
            segment_duration=segment_duration,
            sample_rate=sample_rate,
            channels=channels,
            output_format=output_format,
            debug=debug
        )
        
        # 初始化Whisper转写器
        logger.info(f"初始化Whisper转写器: {whisper_model}")
        transcriber = FasterWhisperTranscribe(
            whisper_model=whisper_model,
            device=device,
            compute_type=compute_type,
            debug=debug
        )
        
        # 初始化队列和线程控制
        segment_queue = queue.Queue(maxsize=max_queue_size)
        stop_event = threading.Event()
        
        # 转写结果
        segments = []
        current_segment_index = 0
        
        # 定义音频流处理线程
        def stream_worker():
            try:
                logger.info("启动音频流处理线程")
                
                # 获取音频段迭代器
                segment_generator = audio_processor.start()
                
                # 处理音频段
                for segment_file in segment_generator:
                    if stop_event.is_set():
                        break
                    
                    try:
                        # 将音频段放入队列
                        logger.debug(f"将音频段放入队列: {segment_file}")
                        segment_queue.put(segment_file, timeout=1)
                    except queue.Full:
                        logger.warning("音频段队列已满，丢弃当前段")
                
                logger.info("音频流处理线程已结束")
                
            except Exception as e:
                logger.error(f"音频流处理线程异常: {str(e)}")
                if debug:
                    logger.error(traceback.format_exc())
                stop_event.set()
        
        # 定义转写线程
        def transcribe_worker():
            nonlocal current_segment_index
            
            try:
                logger.info("启动转写线程")
                
                while not stop_event.is_set():
                    try:
                        # 从队列获取音频段
                        segment_file = segment_queue.get(timeout=1)
                        
                        if segment_file is None:
                            continue
                        
                        # 转写音频段
                        logger.debug(f"转写音频段: {segment_file}")
                        
                        # 记录开始时间
                        start_time = time.time()
                        
                        # 转写音频段
                        segment_results = list(transcriber.transcribe_file_direct(segment_file, language))
                        
                        # 记录结束时间
                        end_time = time.time()
                        processing_time = end_time - start_time
                        
                        # 如果没有结果，跳过
                        if not segment_results:
                            logger.debug(f"音频段无转写结果: {segment_file}")
                            segment_queue.task_done()
                            continue
                        
                        # 处理转写结果
                        for result in segment_results:
                            # 添加段索引
                            result["segment_index"] = current_segment_index
                            result["segment_file"] = segment_file
                            result["processing_time"] = processing_time
                            
                            # 添加到结果列表
                            segments.append(result)
                            
                            # 输出转写结果
                            print(f"[{result['start']:.2f}s - {result['end']:.2f}s]: {result['text']}")
                        
                        # 更新段索引
                        current_segment_index += 1
                        
                        # 标记任务完成
                        segment_queue.task_done()
                        
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"转写音频段异常: {str(e)}")
                        if debug:
                            logger.error(traceback.format_exc())
                
                logger.info("转写线程已结束")
                
            except Exception as e:
                logger.error(f"转写线程异常: {str(e)}")
                if debug:
                    logger.error(traceback.format_exc())
                stop_event.set()
        
        # 启动线程
        stream_thread = threading.Thread(target=stream_worker)
        stream_thread.daemon = True
        stream_thread.start()
        
        transcribe_thread = threading.Thread(target=transcribe_worker)
        transcribe_thread.daemon = True
        transcribe_thread.start()
        
        print(f"开始转写音频流: {stream_url}")
        print("按Ctrl+C停止转写")
        
        # 主循环
        try:
            while True:
                time.sleep(0.1)  # 短暂休眠，避免CPU占用过高
                
        except KeyboardInterrupt:
            print("\n用户中断，停止转写...")
        
        finally:
            # 停止处理
            stop_event.set()
            
            # 停止音频处理器
            audio_processor.stop()
            
            # 等待线程结束
            if stream_thread.is_alive():
                stream_thread.join(timeout=5)
            
            if transcribe_thread.is_alive():
                transcribe_thread.join(timeout=5)
            
            # 输出完整转写结果
            print("\n完整转写结果:")
            full_text = " ".join([segment["text"] for segment in segments])
            print(full_text)
            
            # 清理资源
            audio_processor.cleanup()
            
            print("转写已完成，资源已清理")
        
    except Exception as e:
        print(f"处理异常: {str(e)}")
        if debug:
            traceback.print_exc()


if __name__ == "__main__":
    main()
