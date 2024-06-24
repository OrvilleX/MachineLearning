import cv2
import numpy as np

# 加载图像
image = cv2.imread('path_to_your_image.jpg')
# 转换到灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 或者转换到HSV空间，使用V分量作为亮度信息
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
v_channel = hsv_image[:,:,2]


# 计算平均亮度
average_brightness = np.mean(gray_image)

# 计算亮度直方图
hist = cv2.calcHist([v_channel], [0], None, [256], [0,256])
# 定义亮度异常的阈值
brightness_threshold_high = 240
brightness_threshold_low = 15

if average_brightness > brightness_threshold_high:
    print("图像过亮")
elif average_brightness < brightness_threshold_low:
    print("图像过暗")
else:
    print("亮度正常")

# 显示图像
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 显示直方图
import matplotlib.pyplot as plt
plt.plot(hist)
plt.show()
