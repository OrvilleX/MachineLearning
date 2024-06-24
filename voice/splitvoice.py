from pydub import AudioSegment
from pydub.silence import split_on_silence
import os


def split_long_chunks(chunks, max_length=30000):
    """将音频块列表中超过指定长度的块进一步切分为较小的块。
    Args:
        chunks (List[AudioSegment]): 音频块列表。
        max_length (int): 最大长度，单位毫秒。

    Returns:
        List[AudioSegment]: 切分后的音频块列表。
    """
    output_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_length:
            output_chunks.append(chunk)
        else:
            # 需要切分长音频块
            # 切分为长度为max_length的多个部分
            for start in range(0, len(chunk), max_length):
                end = min(start + max_length, len(chunk))
                output_chunks.append(chunk[start:end])
    return output_chunks


file_path = "path/to/your/audio/file.mp3"
chunk_path = "/root/autodl-tmp/94"
os.makedirs(chunk_path, exist_ok=True)

# 加载音频文件
audio = AudioSegment.from_file(file_path)

# 切分音频
chunks = split_on_silence(
    audio,
    min_silence_len=500,  # 静音时长至少500毫秒
    silence_thresh=-40,  # 小于-40 dBFS以下的为静音
    keep_silence=200  # 在切割后的音频前后各保留200毫秒的静音
)

# 对长于30秒的音频片段进一步切分
chunks = split_long_chunks(chunks, max_length=30000)  # 30秒 = 30000毫秒

# 处理并保存切分后的音频片段
for i, chunk in enumerate(chunks):
    out_file = f"chunk{i}.wav"
    out_file = os.path.join(chunk_path, out_file)
    print(f"Exporting {out_file}")
    chunk.export(out_file, format="wav")
