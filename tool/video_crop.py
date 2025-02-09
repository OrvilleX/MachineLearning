import argparse
import cv2
import os
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
import torch


def check_floats_in_list(float_list, lower_bound=None, upper_bound=None):
    """
    检查float_list中是否存在小于lower_bound或大于upper_bound的元素。

    :param float_list: 浮点数列表
    :param lower_bound: 下界（可选）
    :param upper_bound: 上界（可选）
    :return: 布尔值，表示是否存在满足条件的元素
    """
    if lower_bound is not None:
        if any(x < lower_bound for x in float_list):
            return True

    if upper_bound is not None:
        if any(x > upper_bound for x in float_list):
            return True

    return False


class VideoCrop:
    def __init__(self, video_path, save_path, frame_skip=5, model_path='yolov8.pt', conf_threshold=0.5, method='skip'):
        """
        :param video_path: Path to the video file.
        :param save_path: Path to save the cropped images.
        :param frame_skip: Number of frames to skip for 'skip' method.
        :param model_path: Path to the YOLOv8 model.
        :param conf_threshold: Confidence threshold for YOLOv8 detections.
        :param method: Method to use for cropping images ('skip' or 'yolo').
        """
        self.video_path = video_path
        self.save_path = save_path
        self.frame_skip = frame_skip
        self.model = YOLO(model_path)

        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        self.model.to(torch.device(device))

        self.conf_threshold = conf_threshold
        self.method = method

        # Create save directory if not exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(self.video_path).split('.')[0]

        frame_idx = 0
        saved_count = 0

        with (tqdm(total=frame_count) as pbar):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if self.method == 'skip':
                    # Process every nth frame
                    if frame_idx % self.frame_skip == 0:
                        self.save_frame(frame, basename, saved_count)
                        saved_count += 1

                elif self.method == 'yolo':
                    if self.conf_threshold > 0:
                        results = self.model(frame, conf=self.conf_threshold, verbose=False)
                        confs = results[0].boxes.conf.cpu().tolist()
                        if check_floats_in_list(confs, upper_bound=self.conf_threshold):
                            self.save_frame(frame, basename, saved_count)
                            saved_count += 1
                    else:
                        results = self.model(frame, conf=0.3, classes=[0], verbose=False)
                        confs = results[0].boxes.conf.cpu().tolist()
                        if len(confs) <= 0:
                            # self.save_frame(frame, basename + '_nodet', saved_count)
                            saved_count += 1
                        elif check_floats_in_list(confs, lower_bound=abs(self.conf_threshold)):
                            self.save_frame(frame, basename + '_lowdet', saved_count)
                            saved_count += 1
                frame_idx += 1
                pbar.update(1)

        cap.release()

    def save_frame(self, frame, basename, saved_count):
        save_filename = f"{basename}_{saved_count}.jpg"
        save_filepath = os.path.join(self.save_path, save_filename)
        cv2.imwrite(save_filepath, frame)


if __name__ == '__main__':
    """
    视频帧截取，目前支持多种截取模式，目前模型支持yolo形式的模型
    -s (--source): 视频文件路径
    -o (--output): 图片输出路径
    -m（--mode）: 截取模式，static（固定帧率）,up（模型识别，超过设定阈值截取），down（模型识别，低于设定阈值截取）
    -p（--path）: 运行的推理模型文件路径
    -i (--interval): 多少帧抽一帧
    -c（--conf）: 模型识别的阈值
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type=str, required=True, help='mp4 file path')
    parser.add_argument('-o', '--output', type=str, required=True, help='image save path')
    parser.add_argument('-m', '--mode', type=str, choices=['static', 'yolo'], required=True, help='mode choice')
    parser.add_argument('-p', '--path', type=str, required=False, default='yolov8m.pt', help='model path')
    parser.add_argument('-i', '--interval', type=int, default=3, help='interval second')
    parser.add_argument('-c', '--conf', type=float, default=0.5, help='threshold')
    opt = parser.parse_args()

    if os.path.isfile(opt.source):
        vc = VideoCrop(opt.source, opt.output, opt.interval, opt.path, opt.conf,
                       'skip' if opt.mode == 'static' else 'yolo')
        vc.process_video()
    else:
        files = os.listdir(opt.source)
        files.sort()

        for file_ in files:
            video_path = opt.source + '/' + file_
            vc = VideoCrop(video_path, opt.output, opt.interval, opt.path, opt.conf,
                           'skip' if opt.mode == 'static' else 'yolo')
            vc.process_video()
