import os
import cv2
import argparse
import torch
import numpy as np
from collections import defaultdict
from tracker.yolo_track import YoloTrack

if __name__ == '__main__':
    """
    实现针对各类跟踪的测试验证
    -m（--model）模型文件路径
    -conf（--confidence）上述模型的置信度
    -f（--file）用于测试的视频路径
    -t（--tracker）跟踪的配置文件
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='yolov8n.pt')
    parser.add_argument('-conf', '--confidence', type=float, default=0.5)
    parser.add_argument('-f', '--file', type=str, default='test.mp4')
    parser.add_argument('-t', '--tracker', type=str, default='../conf/bytetrack.yaml')
    opt = parser.parse_args()

    device = 'cpu'
    track_history = defaultdict(lambda: [])
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'

    print(f"current available device is {device}")

    track = YoloTrack(opt.confidence, opt.model, opt.tracker, device)

    videoCapture = cv2.VideoCapture(opt.file)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (1920, 1080)
    frame_idx = 0

    success, frame = videoCapture.read()
    while success:
        success, frame = videoCapture.read()  # 获取下一帧
        frame_idx += 1

        if frame_idx % 4 != 0:
            continue

        results = track.predict(frame[..., ::-1])
        for item in results:
            track_item = track_history[item.track_id]
            track_item.append((float(item.center_x), float(item.center_y)))  # x, y center point
            if len(track_item) > 30:  # retain 90 tracks for 90 frames
                track_item.pop(0)

            # Draw the tracking lines
            points = np.hstack(track_item).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                frame,
                [points],
                isClosed=False,
                color=(230, 230, 230),
                thickness=10,
            )

            cv2.rectangle(frame,
                          (item.x_min, item.y_max),
                          (item.x_max, item.y_min),
                          (0, 0, 255), 2)

            label = f"track:{item.track_id}: {item.conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (item.x_min, item.y_min - h - 10), (item.x_min + w, item.y_min), (0, 255, 0), -1)
            cv2.putText(frame, label, (item.x_min, item.y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Track', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    videoCapture.release()
    cv2.destroyAllWindows()
    print("Finished!")
