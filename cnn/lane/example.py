import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def hough_line():
    """
    Hough_line直线检测算法
    :return:
    """
    img = cv2.imread('../../data/lane/lane2.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 300)
    for i in range(len(lines)):
        r, theta = lines[i, 0, 0], lines[i, 0, 1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("result", img)


def hougnp_line():
    """
    HoughP_line直线检测算法
    :return:
    """
    img = cv2.imread('../../data/lane/lane2.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, 50, 10)
    for i in range(len(lines)):
        cv2.line(img, (lines[i, 0, 0], lines[i, 0, 1]), (lines[i, 0, 2], lines[i, 0, 3]), (0, 255, 0), 2)
    cv2.imshow("result", img)


def lsd_line():
    """
    LSD直线检测算法
    :return:
    """
    img = cv2.imread('../../data/lane/lane2.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(0)
    dlines = lsd.detect(gray)

    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("result", img)


def detect_line_with_ransac():
    # 读取图像并转换为灰度图
    image = cv2.imread('../../data/lane/lane2.png', cv2.IMREAD_GRAYSCALE)

    # 应用Canny边缘检测
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # 获取边缘点的坐标
    points = np.argwhere(edges != 0)
    points = points[:, [1, 0]]  # 转换为(x, y)坐标

    # 使用RANSAC算法拟合直线
    ransac = make_pipeline(PolynomialFeatures(1), RANSACRegressor())
    ransac.fit(points[:, 0][:, np.newaxis], points[:, 1])

    # 获取拟合的直线参数
    inlier_mask = ransac.named_steps['ransacregressor'].inlier_mask_
    line_X = np.arange(points[:, 0].min(), points[:, 0].max())[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)

    # 绘制检测到的直线
    line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # for point, is_inlier in zip(points, inlier_mask):
    #     color = (0, 255, 0) if is_inlier else (255, 0, 0)
    #     cv2.circle(line_image, tuple(point), 2, color, -1)
    for x, y in zip(line_X, line_y_ransac):
        cv2.circle(line_image, (int(x), int(y)), 2, (0, 0, 255), -1)

    # 显示结果
    cv2.imshow('Detected Line with RANSAC', line_image)


def fld_line():
    """

    :return:
    """
    img = cv2.imread('../../data/lane/lane2.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fld = cv2.ximgproc.createFastLineDetector()
    dlines = fld.detect(gray)
    for dline in dlines:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("result", img)
    # drawn_img = fld.drawSegments(img, dlines)
    # cv2.imshow("result", drawn_img)


if __name__ == '__main__':
    # hough_line()
    # hougnp_line()
    # lsd_line()
    # fld_line()
    detect_line_with_ransac()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
