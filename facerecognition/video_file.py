import face_recognition
import cv2
import os
import numpy as np

"""
基于face_recognition实现从视频文件中识别人脸，并匹配人脸以实现从视频中分析存在匹配的人脸
"""


def load_known_face_encodings(knownFaceDir):
    """
    根据指定文件夹读取已知人脸特征
    :param knownFaceDir: 待读取的人脸
    :return: 人脸特征以及对应标识
    """
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(knownFaceDir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(knownFaceDir, filename)
            img = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(img)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(os.path.splitext(filename)[0])
    return known_face_encodings, known_face_names


def read_face_from_video(videoPath, knownFaceDir, faster=True, debug=False):
    """
    从视频文件读取并实现匹配其中的人脸
    :param videoPath: 视频文件地址
    :param knownFaceDir: 已知的人脸文件夹
    :param faster:是否加速识别，即将图片缩小1/4
    :param debug:是否为调试模式，调试模式下仅输出实时的画面，默认不开启调试模式
    :return: 视频中匹配的人脸图片，标识
    """
    video_capture = cv2.VideoCapture(videoPath)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_file = 'output_video.mp4'
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (width, height), isColor=True)

    img_with_name = {}
    known_face_encodings, known_face_names = load_known_face_encodings(knownFaceDir)
    while video_capture.isOpened():
        ret, frame = video_capture.read()

        # 将图片尺寸缩小1/4，以加快整体识别的效率，如果识别准确率不足可以调整
        if faster:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        else:
            small_frame = frame
        rgb_small_frame = small_frame[:, :, ::-1].copy()
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # 选择匹配度度最高的一个作为结果
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # 因为此前缩小了，这里等比放大
            if faster:
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
            if debug:
                cv2.rectangle(rgb_small_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(rgb_small_frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(rgb_small_frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            else:
                crop_img = rgb_small_frame[top:bottom, left:right]
                img_with_name[name] = crop_img
        out.write(rgb_small_frame[:, :, ::-1])
        cv2.imshow('Video', rgb_small_frame[:, :, ::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    out.release()
    return img_with_name


if __name__ == "__main__":
    video_path = "/Users/apple/hszb/video/hszb003.mp4"
    known_face_dir = "/Users/apple/hszb/known_face"
    read_face_from_video(video_path, known_face_dir, faster=False, debug=True)
