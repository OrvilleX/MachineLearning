import cv2
import numpy as np
from utils.image_utils import preprocess_image
from core.detector import GaugeDetector
from core.calculator import GaugeCalculator
from config import Config

def process_gauge(image_path):
    """处理截取后的仪表盘图像并返回读数"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图像")
    
    # 预处理
    original, gray = preprocess_image(image)
    
    edges = cv2.Canny(gray, Config.CANNY_THRESH_1, Config.CANNY_THRESH_2)
    
    # 由于表盘已通过YOLO截取，这里直接根据图像尺寸定义中心与半径
    height, width = original.shape[:2]
    center = (width // 2, height // 2)
    radius = min(width, height) // 2 - 10  # 保留一定边距
    
    # 初始化检测器和计算器
    detector = GaugeDetector()
    detector.center = center
    detector.radius = radius
    
    calculator = GaugeCalculator()
    
    # 检测刻度线
    scale_lines = detector.detect_scale_lines(edges)
    
    # 等待查看HoughLinesP结果
    cv2.waitKey(0)
    cv2.destroyWindow("HoughLinesP Detection")
    
    # 检测数字
    numbers = detector.detect_numbers(gray)
    
    # 检测指针
    pointer = detector.detect_pointer(original)
    if pointer is None:
        raise ValueError("未检测到指针")
    
    # 计算指针角度及实际数值
    angle = calculator.calculate_angle(pointer, detector.center)
    value = calculator.map_angle_to_value(angle, scale_lines, numbers)
    
    # 可视化结果
    result_image = original.copy()
    
    # 绿色圆圈 (0, 255, 0) - 表示表盘边界
    cv2.circle(result_image, center, radius, (0, 255, 0), 2)
    
    # 红色线段 (0, 0, 255) - 表示检测到的刻度线
    for line in scale_lines:
        x1, y1, x2, y2 = line
        cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    # 蓝色轮廓 (255, 0, 0) - 表示检测到的指针
    cv2.drawContours(result_image, [pointer], -1, (255, 0, 0), 2)
    
    # 绿色文字 (0, 255, 0) - 显示读数结果
    cv2.putText(
        result_image,
        f"Value: {value:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    # 可选：显示检测到的数字位置（黄色）
    for number, position in numbers:
        cv2.circle(result_image, position, 3, (0, 255, 255), -1)
        cv2.putText(
            result_image,
            str(number),
            (position[0] + 5, position[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )
    
    # 在返回结果之前等待查看中间结果
    cv2.waitKey(0)
    cv2.destroyWindow("Gray Image")
    cv2.destroyWindow("Blurred Image")
    cv2.destroyWindow("Canny Edges")
    
    return result_image, value

if __name__ == "__main__":
    # 测试代码
    image_path = "../../data/gauge/gauge1.jpg"
    try:
        result_image, value = process_gauge(image_path)
        cv2.imshow("Gauge Reading Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"读数结果: {value}")
    except Exception as e:
        print(f"处理失败: {str(e)}") 