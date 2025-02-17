import numpy as np
import cv2

class GaugeCalculator:
    def __init__(self):
        self.scale_values = {}  # 存储刻度值映射
        
    def calculate_angle(self, pointer_contour, center):
        """计算指针角度"""
        # 计算轮廓的最小外接矩形
        rect = cv2.minAreaRect(pointer_contour)
        angle = rect[2]
        
        # 根据矩形方向调整角度
        if rect[1][0] < rect[1][1]:
            angle += 90
            
        # 确保角度在0-360范围内
        angle = angle % 360
        return angle
    
    def map_angle_to_value(self, angle, scale_lines, numbers):
        """将角度映射到实际值
        Args:
            angle: 指针角度（0-360度）
            scale_lines: 检测到的刻度线列表，每个元素为[x1,y1,x2,y2]
            numbers: 检测到的数字列表，每个元素为(value, position)
        Returns:
            float: 映射后的实际值
        """
        # 如果没有检测到刻度线或数字，使用默认映射
        if not scale_lines or not numbers:
            # 假设表盘范围是0-100，起始角度是-45度，量程是270度
            start_angle = -45
            range_angle = 270
            min_value = 0
            max_value = 100
            
            # 将angle转换为相对于起始角度的角度
            relative_angle = (angle - start_angle) % 360
            if relative_angle > range_angle:
                relative_angle = range_angle
                
            # 线性映射到数值
            value = min_value + (relative_angle / range_angle) * (max_value - min_value)
            return value
            
        # 如果检测到刻度线和数字，建立更精确的映射
        scale_angles = []
        scale_values = []
        
        # 将刻度线转换为角度
        for line in scale_lines:
            x1, y1, x2, y2 = line
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            line_angle = np.degrees(np.arctan2(mid_y - self.center[1], 
                                             mid_x - self.center[0]))
            # 确保角度为正
            if line_angle < 0:
                line_angle += 360
            scale_angles.append(line_angle)
        
        # 将数字位置转换为角度并建立映射
        for value, pos in numbers:
            num_angle = np.degrees(np.arctan2(pos[1] - self.center[1],
                                            pos[0] - self.center[0]))
            if num_angle < 0:
                num_angle += 360
                
            # 找到最近的刻度线
            closest_angle_idx = np.argmin(np.abs(np.array(scale_angles) - num_angle))
            self.scale_values[scale_angles[closest_angle_idx]] = float(value)
        
        # 对已知的刻度值进行排序
        known_angles = sorted(self.scale_values.keys())
        known_values = [self.scale_values[a] for a in known_angles]
        
        # 使用插值计算指针角度对应的值
        for i in range(len(known_angles)-1):
            if known_angles[i] <= angle <= known_angles[i+1]:
                # 线性插值
                ratio = (angle - known_angles[i]) / (known_angles[i+1] - known_angles[i])
                value = known_values[i] + ratio * (known_values[i+1] - known_values[i])
                return value
        
        # 如果角度超出范围，返回最近的值
        if angle < known_angles[0]:
            return known_values[0]
        return known_values[-1] 