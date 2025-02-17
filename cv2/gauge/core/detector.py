import cv2
import numpy as np
import pytesseract
from config import Config

class GaugeDetector:
    def __init__(self):
        self.center = None
        self.radius = None
    
    def visualize_hough_lines(self, edges, lines):
        """可视化HoughLinesP检测到的所有线段
        Args:
            edges: 边缘检测图像
            lines: HoughLinesP检测到的线段
        Returns:
            image: 可视化结果图像
        """
        # 创建彩色图像以显示检测结果
        vis_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 绘制所有检测到的线段（红色）
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        
        # 绘制圆心（绿色）和表盘范围（蓝色）
        cv2.circle(vis_image, self.center, 3, (0, 255, 0), -1)  # 圆心
        cv2.circle(vis_image, self.center, self.radius, (255, 0, 0), 1)  # 表盘范围
        
        return vis_image
    
    def detect_scale_lines(self, edges):
        """检测刻度线
        Args:
            edges: Canny边缘检测后的图像
        Returns:
            list: 刻度线列表，每个元素为[x1,y1,x2,y2]
        """
        lines = cv2.HoughLinesP(
            edges,
            Config.HOUGH_RHO,
            Config.HOUGH_THETA,
            Config.HOUGH_THRESHOLD,
            minLineLength=Config.MIN_LINE_LENGTH,
            maxLineGap=Config.MAX_LINE_GAP
        )
        
        # 保存并显示原始检测结果
        hough_vis = self.visualize_hough_lines(edges, lines)
        cv2.imshow("HoughLinesP Detection", hough_vis)
        cv2.waitKey(1)  # 显示图像，但不阻塞
        
        if lines is None:
            return []
            
        scale_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 计算线段中点
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # 计算线段长度
            line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # 计算线段中点到圆心的距离
            dist_to_center = np.sqrt(
                (mid_x - self.center[0])**2 +
                (mid_y - self.center[1])**2
            )
            
            # 计算线段方向向量
            dx = x2 - x1
            dy = y2 - y1
            
            # 计算从圆心到线段中点的向量
            center_to_line_dx = mid_x - self.center[0]
            center_to_line_dy = mid_y - self.center[1]
            
            # 计算两个向量的夹角（弧度）
            dot_product = dx * center_to_line_dx + dy * center_to_line_dy
            line_mag = np.sqrt(dx**2 + dy**2)
            center_mag = np.sqrt(center_to_line_dx**2 + center_to_line_dy**2)
            
            if line_mag * center_mag == 0:
                continue
            
            cos_angle = dot_product / (line_mag * center_mag)
            cos_angle = min(1, max(-1, cos_angle))  # 确保在[-1,1]范围内
            angle = np.arccos(cos_angle)
            angle_degrees = np.degrees(angle)
            
            # 过滤条件：
            # 1. 线段长度应该在合适范围内（避免太长的外圈或太短的噪声）
            # 2. 线段位置应在圆周附近
            # 3. 线段方向应该接近径向（与圆心连线夹角接近0度或180度）
            if (self.radius * 0.1 < line_length < self.radius * 0.3 and  # 长度为半径的10-30%
                0.8 * self.radius < dist_to_center < 1.2 * self.radius and  # 位置在圆周附近
                (angle_degrees < 20 or angle_degrees > 160)):  # 接近径向方向
                
                # 确保线段方向是从圆心指向外部
                if dist_to_center > 0:
                    # 计算应该的方向
                    desired_dx = center_to_line_dx / center_mag
                    desired_dy = center_to_line_dy / center_mag
                    
                    # 设置新的端点，使线段从圆心指向外部
                    new_length = self.radius * 0.2  # 统一线段长度
                    x1 = int(mid_x - desired_dx * new_length * 0.3)  # 内端点
                    y1 = int(mid_y - desired_dy * new_length * 0.3)
                    x2 = int(mid_x + desired_dx * new_length * 0.7)  # 外端点
                    y2 = int(mid_y + desired_dy * new_length * 0.7)
                    
                    scale_lines.append([x1, y1, x2, y2])
        
        # 过滤重复的刻度线
        filtered_lines = []
        for i, line1 in enumerate(scale_lines):
            is_duplicate = False
            x1, y1, x2, y2 = line1
            for j, line2 in enumerate(filtered_lines):
                x3, y3, x4, y4 = line2
                # 计算线段中点距离
                mid1 = ((x1 + x2)/2, (y1 + y2)/2)
                mid2 = ((x3 + x4)/2, (y3 + y4)/2)
                dist = np.sqrt((mid1[0]-mid2[0])**2 + (mid1[1]-mid2[1])**2)
                if dist < self.radius * 0.05:  # 如果两条线段太近，认为是重复
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_lines.append(line1)
        
        return filtered_lines
    
    def detect_numbers(self, image):
        """检测表盘数字"""
        # 使用Tesseract OCR检测数字
        numbers = []
        # 在圆周附近提取ROI进行OCR
        # ... (根据实际需求实现数字检测逻辑)
        return numbers
    
    def detect_pointer(self, image):
        """检测指针"""
        # 使用形态学操作和轮廓检测找到指针
        mask = cv2.inRange(image, (0, 0, 0), (50, 50, 50))
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 找到最长的轮廓作为指针
        if contours:
            pointer = max(contours, key=cv2.contourArea)
            return pointer
        return None
