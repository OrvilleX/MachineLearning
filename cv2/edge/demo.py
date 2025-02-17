import cv2
import numpy as np
import os

def auto_canny(image, sigma=0.33):
    """自适应 Canny 边缘检测
    Args:
        image: 输入图像
        sigma: 控制阈值范围的参数，默认0.33
    Returns:
        edges: 边缘检测结果
    """
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))  # 低阈值
    upper = int(min(255, (1.0 + sigma) * v))  # 高阈值
    return cv2.Canny(image, lower, upper)

def apply_edge_detection(image):
    """应用多种边缘检测算法
    Args:
        image: 输入图像
    Returns:
        dict: 包含各种边缘检测结果的字典
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊（减少噪声）
    # 高斯模糊可以有效减少图像中的噪声，使边缘检测结果更干净。核心参数为 ksize，该参数过小将导致噪声
    # 无法有效去除，过大则会导致边缘过于模糊影响识别，并且处理速度也会下降。
    blurred = gray # cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Canny 边缘检测
    # 参数：低阈值 50，高阈值 150
    canny_edges = cv2.Canny(blurred, 30, 120)
    
    # 自适应 Canny 边缘检测
    # 参数：sigma=0.33（自动计算高低阈值）
    auto_canny_edges = auto_canny(blurred)
    
    # Sobel 边缘检测
    # 参数：x方向梯度，y方向梯度，核大小 5
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobelx, sobely)  # 计算梯度幅值
    sobel_edges = np.uint8(sobel_edges)
    
    # Laplacian 边缘检测
    # 参数：输出图像深度 CV_64F
    laplacian_edges = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian_edges = np.uint8(np.absolute(laplacian_edges))  # 取绝对值并转换为8位
    
    # Scharr 边缘检测
    # 参数：x方向梯度，y方向梯度
    scharrx = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)
    scharr_edges = cv2.magnitude(scharrx, scharry)  # 计算梯度幅值
    scharr_edges = np.uint8(scharr_edges)
    
    # Prewitt 边缘检测（通过自定义核实现）
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewittx = cv2.filter2D(blurred, cv2.CV_64F, kernelx)  # 确保输出类型为 CV_64F
    prewitty = cv2.filter2D(blurred, cv2.CV_64F, kernely)  # 确保输出类型为 CV_64F
    prewitt_edges = cv2.magnitude(prewittx, prewitty)
    prewitt_edges = np.uint8(prewitt_edges)
    
    # 对 Sobel 和 Prewitt 结果进行过滤（仅过滤低于阈值，其他值保持不变）
    _, canny_filtered = cv2.threshold(canny_edges, 40, 255, cv2.THRESH_TOZERO)
    _, prewitt_filtered = cv2.threshold(sobel_edges, 10, 255, cv2.THRESH_TOZERO)

    # 计算 canny_edges 与 prewitt_edges 的未重叠部分
    prewitt_unique = cv2.subtract(prewitt_filtered, canny_filtered)

    # 对未重叠部分进行加权融合
    alpha = 0.8  # prewitt_unique 的权重
    beta = 1.0   # canny_edges 的权重
    fused_edges = cv2.addWeighted(canny_filtered, beta, prewitt_unique, alpha, 0)

    # 转换为 8 位图像
    fused_edges = np.uint8(fused_edges)

    return {
        "Original": fused_edges,  # 替换 Original 为增强后的结果
        "Gray": gray,
        "Canny": canny_edges,
        "Auto Canny": auto_canny_edges,
        "Sobel": sobel_edges,
        "Laplacian": laplacian_edges,
        "Scharr": scharr_edges,
        "Prewitt": prewitt_edges
    }

def display_results(results):
    """在一个窗口中显示所有结果
    Args:
        results: 包含各种边缘检测结果的字典
    """
    # 将所有结果拼接成一个大图
    row1 = np.hstack([cv2.cvtColor(results["Original"], cv2.COLOR_GRAY2BGR), 
                      cv2.cvtColor(results["Gray"], cv2.COLOR_GRAY2BGR)])
    row2 = np.hstack([cv2.cvtColor(results["Canny"], cv2.COLOR_GRAY2BGR), 
                      cv2.cvtColor(results["Auto Canny"], cv2.COLOR_GRAY2BGR)])
    row3 = np.hstack([cv2.cvtColor(results["Sobel"], cv2.COLOR_GRAY2BGR), 
                      cv2.cvtColor(results["Laplacian"], cv2.COLOR_GRAY2BGR)])
    row4 = np.hstack([cv2.cvtColor(results["Scharr"], cv2.COLOR_GRAY2BGR), 
                      cv2.cvtColor(results["Prewitt"], cv2.COLOR_GRAY2BGR)])
    
    combined = np.vstack([row1, row2, row3, row4])
    
    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Fused Edges", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Gray", (results["Original"].shape[1] + 10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Canny", (10, results["Original"].shape[0] + 30), font, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Auto Canny", (results["Original"].shape[1] + 10, results["Original"].shape[0] + 30), font, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Sobel", (10, 2 * results["Original"].shape[0] + 30), font, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Laplacian", (results["Original"].shape[1] + 10, 2 * results["Original"].shape[0] + 30), font, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Scharr", (10, 3 * results["Original"].shape[0] + 30), font, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Prewitt", (results["Original"].shape[1] + 10, 3 * results["Original"].shape[0] + 30), font, 1, (0, 255, 0), 2)
    
    # 显示结果
    cv2.imshow("Edge Detection Comparison", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 图片路径数组
    image_paths = [
        "../../data/gauge/gauge1.jpg",
        "../../data/gauge/gauge2.png",
        "../../data/gauge/gauge3.png",
        "../../data/gauge/gauge4.jpg",
        "../../data/gauge/gauge5.png",
        "../../data/gauge/gauge6.jpg",
        "../../data/gauge/gauge7.jpg",
        "../../data/gauge/gauge8.png",
        "../../data/gauge/gauge9.png"
    ]

    for image_path in image_paths:
        # 读取图像
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"无法读取图像：{image_path}，请检查路径是否正确")
            continue

        # 应用边缘检测算法
        results = apply_edge_detection(image)
        
        # 获取融合后的边缘检测结果
        fused_edges = results["Original"]

        # 生成保存路径
        dir_path, file_name = os.path.split(image_path)
        save_path = os.path.join(dir_path, f"fused_{file_name}")

        # 保存结果
        cv2.imwrite(save_path, fused_edges)
        print(f"已保存结果到：{save_path}")
