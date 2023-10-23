import cv2
import numpy as np


def hough_line():
    """
    Hough_line直线检测算法
    :return:
    """
    img = cv2.imread('../../data/lane/lane2.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 300)
    for i in range(len(lines)):
        r, theta = lines[i, 0, 0], lines[i, 0, 1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*r
        y0 = b*r
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*a)
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*a)
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
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, 50, 10)
    for i in range(len(lines)):
        cv2.line(img, (lines[i, 0, 0], lines[i, 0, 1]), (lines[i, 0,2], lines[i, 0, 3]), (0, 255, 0), 2)
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
    cv2.waitKey(0)