import os
import time
import threading
import queue
import logging
import tempfile
import traceback
import subprocess
import shutil
from typing import Iterator, Optional
import ffmpeg

class AudioStreamProcessor:
    """音频流处理器"""
    
    def __init__(self, 
                 stream_url: str,
                 segment_duration: int = 10,
                 sample_rate: int = 16000,
                 channels: int = 1,
                 output_format: str = "wav",
                 debug: bool = False,
                 debug_dir: str = "./debug_audio",
                 log_level: int = logging.INFO,
                 max_retries: int = 3,
                 retry_delay: int = 5,
                 ffmpeg_loglevel: str = "warning"):
        """
        初始化音频流处理器
        
        Args:
            stream_url: 音频流URL (支持http/https/rtsp/rtmp)
            segment_duration: 每个音频段的持续时间（秒）
            sample_rate: 采样率（Hz）
            channels: 音频通道数
            output_format: 输出格式 (wav/flac)
            debug: 是否启用调试模式
            debug_dir: 调试模式下保存音频文件的目录
            log_level: 日志级别
            max_retries: 连接失败时的最大重试次数
            retry_delay: 重试之间的延迟（秒）
            ffmpeg_loglevel: ffmpeg日志级别
        """
        # 基本配置
        self.stream_url = stream_url
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.channels = channels
        self.output_format = output_format.lower()
        
        # 确保输出格式有效
        if self.output_format not in ["wav", "flac"]:
            raise ValueError("输出格式必须是 'wav' 或 'flac'")
        
        # 调试配置
        self.debug = debug
        self.debug_dir = debug_dir
        if self.debug and not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir, exist_ok=True)
        
        # 临时目录配置
        self.temp_dir = tempfile.mkdtemp(prefix="audio_stream_")
        
        # 重试配置
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.ffmpeg_loglevel = ffmpeg_loglevel
        
        # 设置日志
        self.logger = self._setup_logger(log_level)
        
        # 运行状态
        self.is_running = False
        self.is_paused = False
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.segment_queue = queue.Queue()
        self.segment_index = 0
        self.ffmpeg_process = None
        
        # 检查ffmpeg是否可用
        self._check_ffmpeg()
        
        self.logger.info(f"初始化音频流处理器: {stream_url}")
        self.logger.info(f"音频格式: {channels}通道, {sample_rate}Hz, 输出格式: {output_format}")

    def _setup_logger(self, log_level: int) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("AudioStreamProcessor")
        logger.setLevel(log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _check_ffmpeg(self):
        """检查ffmpeg是否可用"""
        try:
            subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.logger.debug("ffmpeg可用")
        except Exception as e:
            self.logger.error("ffmpeg不可用，请确保已安装ffmpeg并添加到PATH中")
            raise RuntimeError("ffmpeg不可用") from e

    def start(self) -> Iterator[str]:
        """
        开始处理音频流并返回音频段文件路径的迭代器
        
        Returns:
            Iterator[str]: 音频段文件路径的迭代器
        """
        if self.is_running:
            self.logger.warning("处理器已在运行中")
            return
        
        self.is_running = True
        self.is_paused = False
        self.stop_event.clear()
        self.pause_event.clear()
        
        process_thread = threading.Thread(target=self._process_stream)
        process_thread.daemon = True
        process_thread.start()
        
        try:
            while not self.stop_event.is_set() or not self.segment_queue.empty():
                try:
                    segment_file = self.segment_queue.get(timeout=1)
                    if segment_file is None:
                        break
                    yield segment_file
                except queue.Empty:
                    if not process_thread.is_alive() and self.is_running:
                        self.logger.warning("处理线程已结束，但处理器仍在运行状态")
                        break
                    continue
                
        except KeyboardInterrupt:
            self.logger.info("用户中断，停止处理...")
            self.stop()
        
        finally:
            if process_thread.is_alive():
                process_thread.join(timeout=5)
            self.is_running = False
            self.logger.info("音频流处理已完成")

    def pause(self):
        """暂停处理音频流"""
        if not self.is_running:
            self.logger.warning("处理器未运行，无法暂停")
            return
        
        if self.is_paused:
            self.logger.warning("处理器已经处于暂停状态")
            return
        
        self.logger.info("暂停音频流处理")
        self.is_paused = True
        self.pause_event.set()
        
        # 如果ffmpeg进程正在运行，终止它
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait(timeout=5)
            self.ffmpeg_process = None

    def resume(self):
        """恢复处理音频流"""
        if not self.is_running:
            self.logger.warning("处理器未运行，无法恢复")
            return
        
        if not self.is_paused:
            self.logger.warning("处理器未处于暂停状态")
            return
        
        self.logger.info("恢复音频流处理")
        self.is_paused = False
        self.pause_event.clear()

    def stop(self):
        """停止处理音频流"""
        if not self.is_running:
            self.logger.warning("处理器未运行")
            return
        
        self.logger.info("停止音频流处理")
        self.stop_event.set()
        self.pause_event.clear()
        
        # 如果ffmpeg进程正在运行，终止它
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait(timeout=5)
            self.ffmpeg_process = None
        
        # 确保队列中有一个None，以便迭代器结束
        self.segment_queue.put(None)

    def _process_stream(self):
        """处理音频流的主线程"""
        self.logger.info(f"开始处理音频流: {self.stream_url}")
        
        retry_count = 0
        while not self.stop_event.is_set() and retry_count <= self.max_retries:
            try:
                if self.is_paused:
                    self.logger.info("处理器已暂停，等待恢复...")
                    while self.is_paused and not self.stop_event.is_set():
                        time.sleep(0.5)
                    if self.stop_event.is_set():
                        break
                    self.logger.info("处理器已恢复")
                
                # 创建分段目录
                segment_dir = os.path.join(self.temp_dir, f"segments_{int(time.time())}")
                os.makedirs(segment_dir, exist_ok=True)
                
                # 构建ffmpeg命令
                segment_pattern = os.path.join(segment_dir, f"segment_%06d.{self.output_format}")
                
                # 使用ffmpeg-python库构建命令
                stream = ffmpeg.input(self.stream_url, loglevel=self.ffmpeg_loglevel)
                stream = ffmpeg.output(
                    stream,
                    segment_pattern,
                    ac=self.channels,
                    ar=self.sample_rate,
                    segment_time=self.segment_duration,
                    f='segment'
                )
                
                # 启动ffmpeg进程
                self.logger.info(f"启动ffmpeg进程，分段模式，每段{self.segment_duration}秒")
                self.ffmpeg_process = ffmpeg.run_async(stream, pipe_stdout=True, pipe_stderr=True)
                
                # 启动文件监控线程
                monitor_thread = threading.Thread(
                    target=self._monitor_segments, 
                    args=(segment_dir,)
                )
                monitor_thread.daemon = True
                monitor_thread.start()
                
                # 等待ffmpeg进程结束
                returncode = self.ffmpeg_process.wait()
                
                if returncode != 0 and not self.stop_event.is_set() and not self.is_paused:
                    stderr = self.ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                    self.logger.error(f"ffmpeg进程异常退出，返回码: {returncode}")
                    self.logger.error(f"ffmpeg错误输出: {stderr}")
                    raise Exception(f"ffmpeg进程异常退出，返回码: {returncode}")
                
                # 等待监控线程结束
                monitor_thread.join(timeout=5)
                
                # 如果是正常结束，不再重试
                if not self.stop_event.is_set() and not self.is_paused:
                    self.logger.info("音频流已结束")
                break
                
            except Exception as e:
                if self.stop_event.is_set() or self.is_paused:
                    break
                
                retry_count += 1
                self.logger.error(f"处理流失败 (尝试 {retry_count}/{self.max_retries}): {str(e)}")
                
                if retry_count <= self.max_retries:
                    self.logger.info(f"将在 {self.retry_delay} 秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"达到最大重试次数 ({self.max_retries})，放弃处理")
        
        # 确保队列中有一个None，以便迭代器结束
        self.segment_queue.put(None)
        self.logger.info("处理线程已结束")

    def _monitor_segments(self, segment_dir: str):
        """监控分段目录，将新生成的分段文件添加到队列中"""
        self.logger.info(f"开始监控分段目录: {segment_dir}")
        
        processed_files = set()
        
        while not self.stop_event.is_set() and not self.is_paused:
            try:
                # 获取目录中的所有文件
                files = sorted([f for f in os.listdir(segment_dir) 
                               if f.endswith(f".{self.output_format}")])
                
                # 处理新文件
                for file in files:
                    file_path = os.path.join(segment_dir, file)
                    
                    # 如果文件已处理，跳过
                    if file_path in processed_files:
                        continue
                    
                    # 等待文件写入完成
                    if self._is_file_ready(file_path):
                        # 处理文件
                        output_path = self._process_segment(file_path)
                        if output_path:
                            self.segment_queue.put(output_path)
                            processed_files.add(file_path)
                
                # 检查ffmpeg进程是否仍在运行
                if self.ffmpeg_process and self.ffmpeg_process.poll() is not None:
                    # 处理最后可能的文件
                    for file in sorted([f for f in os.listdir(segment_dir) 
                                      if f.endswith(f".{self.output_format}")]):
                        file_path = os.path.join(segment_dir, file)
                        if file_path not in processed_files and self._is_file_ready(file_path):
                            output_path = self._process_segment(file_path)
                            if output_path:
                                self.segment_queue.put(output_path)
                                processed_files.add(file_path)
                    break
                
                time.sleep(0.1)  # 短暂休眠，避免CPU占用过高
                
            except Exception as e:
                self.logger.error(f"监控分段目录时出错: {str(e)}")
                self.logger.error(traceback.format_exc())
                time.sleep(1)  # 出错时稍长的休眠
        
        self.logger.info(f"停止监控分段目录: {segment_dir}")

    def _is_file_ready(self, file_path: str) -> bool:
        """
        检查文件是否已经写入完成
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 文件是否已经写入完成
        """
        if not os.path.exists(file_path):
            return False
        
        try:
            # 尝试获取文件大小
            size1 = os.path.getsize(file_path)
            time.sleep(0.1)  # 等待一小段时间
            size2 = os.path.getsize(file_path)
            
            # 如果文件大小没有变化，认为文件已经写入完成
            return size1 == size2 and size1 > 0
        except Exception:
            return False

    def _process_segment(self, file_path: str) -> Optional[str]:
        """
        处理音频分段，确保与faster-whisper兼容
        
        Args:
            file_path: 分段文件路径
            
        Returns:
            Optional[str]: 处理后的文件路径，如果处理失败则返回None
        """
        try:
            # 生成输出文件路径
            output_path = os.path.join(
                self.temp_dir, 
                f"processed_segment_{self.segment_index:06d}.{self.output_format}"
            )
            
            # 使用ffmpeg确保音频格式正确
            stream = ffmpeg.input(file_path)
            stream = ffmpeg.output(
                stream,
                output_path,
                format=self.output_format,
                ac=self.channels,
                ar=self.sample_rate,
                loglevel=self.ffmpeg_loglevel
            )
            ffmpeg.run(stream, quiet=True, overwrite_output=True)
            
            # 如果启用调试模式，也保存一份到调试目录
            if self.debug:
                debug_path = os.path.join(
                    self.debug_dir, 
                    f"segment_{self.segment_index:06d}.{self.output_format}"
                )
                shutil.copy2(output_path, debug_path)
            
            self.logger.info(f"处理分段 {self.segment_index}: {output_path}")
            self.segment_index += 1
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"处理分段失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def cleanup(self):
        """清理临时文件"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"已清理临时目录: {self.temp_dir}")
        except Exception as e:
            self.logger.error(f"清理临时文件失败: {str(e)}")

    def __del__(self):
        """析构函数，确保资源被释放"""
        self.stop()
        self.cleanup()


# 使用示例
if __name__ == "__main__":
    # 直接定义参数变量
    url = "http://localhost:6006/stream"  # 音频流URL (支持http/https/rtsp/rtmp)
    segment_duration = 10  # 每个音频段的持续时间（秒）
    sample_rate = 16000  # 采样率（Hz）
    channels = 1  # 音频通道数
    output_format = "wav"  # 输出格式 (wav/flac)
    debug = True  # 启用调试模式
    debug_dir = "./debug_audio"  # 调试模式下保存音频文件的目录
    log_level = logging.INFO  # 日志级别
    max_retries = 3  # 连接失败时的最大重试次数
    retry_delay = 5  # 重试之间的延迟（秒）
    
    # 创建处理器
    processor = AudioStreamProcessor(
        stream_url=url,
        segment_duration=segment_duration,
        sample_rate=sample_rate,
        channels=channels,
        output_format=output_format,
        debug=debug,
        debug_dir=debug_dir,
        log_level=log_level,
        max_retries=max_retries,
        retry_delay=retry_delay
    )
    
    try:
        print(f"开始处理音频流: {url}")
        print("等待音频段生成，按Ctrl+C停止...")
        
        # 启动处理器并获取音频段迭代器
        segment_generator = processor.start()
        
        # 设置超时计数器
        timeout_counter = 0
        max_timeout = 60  # 最大等待时间（秒）
        
        # 简单循环处理音频段
        while True:
            try:
                # 尝试获取下一个音频段
                segment_file = next(segment_generator)
                
                # 重置超时计数器
                timeout_counter = 0
                
                # 输出音频段文件路径
                print(f"生成音频段: {segment_file}")
                print(f"文件大小: {os.path.getsize(segment_file)} 字节")
                print("-" * 80)
                
            except StopIteration:
                # 检查是否超时
                timeout_counter += 1
                if timeout_counter >= max_timeout:
                    print(f"已等待 {max_timeout} 秒未收到音频段，退出程序")
                    break
                
                # 每10秒输出一次等待信息，避免过多输出
                if timeout_counter % 10 == 0:
                    print(f"等待音频段... ({timeout_counter}/{max_timeout}秒)")
                
                time.sleep(1)  # 等待1秒后再次尝试
            
            except Exception as e:
                print(f"处理音频段时出错: {str(e)}")
                traceback.print_exc()
                time.sleep(1)  # 出错时稍长的休眠
        
    except KeyboardInterrupt:
        print("\n用户中断，停止处理...")
    
    except Exception as e:
        print(f"处理音频流时发生错误: {str(e)}")
        traceback.print_exc()
    
    finally:
        # 停止处理器并清理资源
        processor.stop()
        processor.cleanup()