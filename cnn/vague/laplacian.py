import matplotlib.pyplot as plt
import numpy as np
import operator
import os

import cv2

if __name__ == '__main__':
    folder_path = '../../data/vague'
    files = os.listdir(folder_path)
    img_blur = {}
    img_sobely = {}
    img_scharr = {}
    for file in files:
        img = cv2.imread(os.path.join(folder_path, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()

        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        sharpness = np.mean(gradient_magnitude)

        scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        gradient_magnitude = np.sqrt(scharrx ** 2 + scharry ** 2)
        scharr_sharpness = np.mean(gradient_magnitude)

        img_blur[file[:-4]] = blur
        img_sobely[file[:-4]] = sharpness
        img_scharr[file[:-4]] = scharr_sharpness
        print('%s Laplacian:%d Sobel:%d Scharr:%d' % (file[:-4], blur, sharpness, scharr_sharpness))

    sorted_img_blur = dict(sorted(img_blur.items(), key=operator.itemgetter(1)))
    sorted_img_sobely = dict(sorted(img_sobely.items(), key=operator.itemgetter(1)))
    scharr_img_canny = dict(sorted(img_scharr.items(), key=operator.itemgetter(1)))

    fig, ax1 = plt.subplots(figsize=(15, 8))

    # 绘制折线图
    ax1.plot(list(sorted_img_blur.keys()), list(sorted_img_blur.values()), 'g-')
    ax1.set_xlabel('图片序号', fontproperties='SimHei')
    ax1.set_ylabel('Laplacian', color='g')
    plt.xticks(rotation=60)
    plt.title('图片模糊度分析', fontproperties='SimHei')

    ax2 = ax1.twinx()
    ax2.plot(list(sorted_img_blur.keys()), list(sorted_img_sobely.values()), 'b-')
    ax2.set_ylabel('Sobel', color='b')

    # ax3 = ax1.twinx()
    # ax3.plot(list(scharr_img_canny.keys()), list(scharr_img_canny.values()), 'r-')
    # ax3.set_ylabel('Scharr', color='r')

    # 显示图表
    plt.show()


