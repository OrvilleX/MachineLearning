import numpy as np

class Config:
    # 图像预处理参数
    RESIZE_WIDTH = 500        # 图像缩放宽度，根据实际图片大小调整
    BLUR_KERNEL_SIZE = (5, 5) # 高斯模糊核大小，越大越模糊，通常为奇数
    CANNY_THRESH_1 = 20       # Canny边缘检测的低阈值
    CANNY_THRESH_2 = 50      # Canny边缘检测的高阈值，通常是低阈值的2-3倍
    
    # 圆检测参数 (HoughCircles)
    CIRCLE_PARAM_1 = 50       # Canny边缘检测的高阈值，越小检测到的圆越多
    CIRCLE_PARAM_2 = 30       # 累加器阈值，越小检测到的圆越多
    MIN_RADIUS = 100          # 最小圆半径（像素）
    MAX_RADIUS = 400          # 最大圆半径（像素）
    
    # 直线检测参数 (HoughLinesP)
    HOUGH_RHO = 1            # 距离分辨率（像素）
    HOUGH_THETA = np.pi/180  # 角度分辨率（弧度）
    HOUGH_THRESHOLD = 40     # 累加器阈值，越大检测的线越少但更准确
    MIN_LINE_LENGTH = 30     # 最小线段长度（像素）
    MAX_LINE_GAP = 10        # 最大线段间隔（像素）
    
    # 指针检测参数
    POINTER_COLOR_LOWER = (0, 0, 0)      # 指针颜色范围下限 (BGR)
    POINTER_COLOR_UPPER = (50, 50, 50)   # 指针颜色范围上限 (BGR)
    POINTER_MIN_AREA = 100               # 指针最小面积（像素）
    POINTER_MAX_AREA = 5000              # 指针最大面积（像素）
    
    # 刻度线过滤参数
    SCALE_LINE_DIST_THRESH = 20          # 刻度线到圆心距离容差（像素）
    SCALE_MIN_ANGLE = 0                  # 刻度最小角度（度）
    SCALE_MAX_ANGLE = 360                # 刻度最大角度（度）
    
    # OCR配置
    DIGIT_MIN_CONFIDENCE = 60            # OCR最小置信度
    DIGIT_MIN_HEIGHT = 20                # 数字最小高度（像素）
    DIGIT_MAX_HEIGHT = 50                # 数字最大高度（像素）
    
    # 数值映射参数
    MIN_VALUE = 0                        # 表盘最小值
    MAX_VALUE = 100                      # 表盘最大值
    VALUE_STEP = 10                      # 刻度值步长 