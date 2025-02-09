import os
import argparse
from ultralytics import YOLO

class YoloTrack:
    def __init__(self, confidence, model_path, tracker_path):
        self.confidenece = confidence
        self.detection_model = YOLO(model_path)
        self.tracker_settings = tracker_path

    def predict(self, frame):
        results = self.detection_model.track(source=frame, persist=True, tracker=settings.personal_track, classes=[0],
                                             conf=self.confidence)
        for result in results:
            for box in result.boxes:
                if box.id is None:
                    break
                track_id = box.id.item()
                x_min = int(box.xyxy[0, 0])
                y_min = int(box.xyxy[0, 1])
                x_max = int(box.xyxy[0, 2])
                y_max = int(box.xyxy[0, 3])
                person_crop_img = frame[int(y_min): int(y_max), int(x_min): int(x_max)]
                self.person_id.append(track_id)
                self.person_box.append((int(x_min), int(y_min), int(x_max), int(y_max)))
                self.person_list.append(person_crop_img)
                self.person_conf.append(float(box.conf[0]))
        if len(self.person_list) > 0:
            return True
        return False


if __name__ == '__main__':
    """
    实现针对yolov8跟踪的测试验证
    -m（--model）yolov8模型文件路径
    -conf（--confidence）上述模型的置信度
    -f（--file）用于测试的视频路径
    -t（--tracker）跟踪的配置文件
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='../images')
    parser.add_argument('-o', '--output', type=str, default='../crop_images')
    parser.add_argument('-m', '--model', type=str, default='yolov5s.pt')
    opt = parser.parse_args()