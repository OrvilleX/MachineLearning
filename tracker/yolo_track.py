import torch
from ultralytics import YOLO
from tracker.track_item import TrackItem


class YoloTrack:
    def __init__(self, confidence, model_path, tracker_path, device='cpu'):
        self.confidence = confidence
        self.detection_model = YOLO(model_path)
        self.detection_model.to(torch.device(device))
        self.tracker_settings = tracker_path

    def predict(self, frame):
        track_list = []
        results = self.detection_model.track(source=frame, persist=True, tracker=self.tracker_settings, classes=[0, 1],
                                             conf=self.confidence, iou=0.8)
        if results[0].boxes.id is None:
            return track_list

        xyboxes = results[0].boxes.xyxy.cpu()
        whboxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confs = results[0].boxes.conf.cpu().tolist()

        for xybox, whbox, track_id, conf in zip(xyboxes, whboxes, track_ids, confs):
            x, y, w, h = whbox
            x_min, y_min, x_max, y_max = xybox
            track_item = TrackItem(track_id, conf=conf)
            track_item.set_xywh(x, y, w, h)
            track_item.set_xyxy(x_min, y_min, x_max, y_max)
            # person_crop_img = frame[int(y_min): int(y_max), int(x_min): int(x_max)]
            track_list.append(track_item)
        return track_list
