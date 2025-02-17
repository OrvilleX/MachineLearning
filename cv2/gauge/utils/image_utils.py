import cv2
from config import Config

def preprocess_image(image):
    """图像预处理：调整大小、灰度化、去噪"""
    # 调整图像大小
    scale = Config.RESIZE_WIDTH / image.shape[1]
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    image = cv2.resize(image, (width, height))
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image, gray
