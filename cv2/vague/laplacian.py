import cv2
import numpy as np

if __name__ == '__main__':
    # 读取图像
    gray_image = cv2.imread('../../data/vague/16.png', cv2.IMREAD_GRAYSCALE)
    image = cv2.imread('../../data/vague/16.png')

    # 将图像分割成RGB通道
    b, g, r = cv2.split(image)

    # 对图像应用高斯滤波
    blurred_image_b = cv2.GaussianBlur(b, (3, 3), 0)
    blurred_image_g = cv2.GaussianBlur(g, (3, 3), 0)
    blurred_image_r = cv2.GaussianBlur(r, (3, 3), 0)

    # 应用拉普拉斯算子
    laplacian_b = cv2.Laplacian(blurred_image_b, cv2.CV_64F, ksize=1)
    laplacian_g = cv2.Laplacian(blurred_image_g, cv2.CV_64F, ksize=1)
    laplacian_r = cv2.Laplacian(blurred_image_r, cv2.CV_64F, ksize=1)

    merge_image = cv2.merge((
        cv2.convertScaleAbs(laplacian_b),
        cv2.convertScaleAbs(laplacian_g),
        cv2.convertScaleAbs(laplacian_r)
    ))

    enhanced_image = cv2.addWeighted(image, 1.0, merge_image, 1.0, 0.0)

    # 显示原始图像、拉普拉斯算子处理后的图像和增强后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Enhanced Image', enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
