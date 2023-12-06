import cv2
import numpy as np


def process_gray():
    # 读取图像并转换为灰度图像
    image = cv2.imread('../../data/vague/16.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Sobel算子计算图像的水平和垂直方向梯度
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # 合并水平和垂直方向的梯度
    combined_sobel = cv2.magnitude(sobel_x, sobel_y)

    # 对梯度图像进行阈值处理
    threshold_value = 50
    thresholded_image = cv2.threshold(combined_sobel, threshold_value, 255, cv2.THRESH_BINARY)[1]

    # 显示增强后的图像
    cv2.imshow('Enhanced Image', thresholded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    process_gray()
