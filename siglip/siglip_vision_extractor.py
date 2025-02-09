"""
siglip_vision_extractor.py

该脚本独立实现了基于 SiglipVisionTower 模型的视频/图像帧特征提取，
同时提供了如何用 matplotlib 进行画布式显示，以及如何将提取的特征传入其他开源 LLM（例如 qwen/deepseek）的示例。

依赖：
    - torch
    - transformers (其中包含 SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig)
    - PIL
    - matplotlib (用于可视化，可选)
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2

# 导入 transformers 中有关 Siglip 模型的组件
from transformers import SiglipImageProcessor, SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionModel

# 从 core 模块引入 VisionTower 基类和配置类
from core.vision_tower import VisionTower, VisionTowerConfig

# -----------------------
# 定义 VisionTower 相关配置和基类
# -----------------------

# -----------------------
# 定义 SiglipVisionTower 模型：继承 VisionTower
# -----------------------

class SiglipVisionTower(VisionTower):
    """
    SiglipVisionTower 封装了 Siglip 对视觉特征提取的核心实现，
    并通过继承 VisionTower 保留了通用的前向传播和特征选择逻辑。
    """
    def __init__(self, model_name_or_path: str, config: PretrainedConfig, vision_config: PretrainedConfig = None):
        # 若未传入 vision_config，则从预训练模型中加载 SiglipVisionConfig
        if vision_config is None:
            vision_config = SiglipVisionConfig.from_pretrained(model_name_or_path)
        super().__init__(model_name_or_path, config, vision_config)
        self.vision_config = vision_config
        self.vision_tower_name = model_name_or_path
        # 使用 transformers 自带的 SiglipImageProcessor 对图像进行预处理
        self.vision_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        print('Loading SiglipVisionTower from:', model_name_or_path)
        # 从预训练模型中加载 SiglipVisionModel
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
        # 根据 vision_config 设置相关尺寸参数
        self.hidden_size = vision_config.hidden_size
        self.W = self.H = vision_config.image_size // vision_config.patch_size
        self.T = 1
        self.select_feature = "cls_patch"

# -----------------------
# 示例：特征提取和向其他 LLM 传递特征
# -----------------------

def extract_features_from_image(image_path, model: SiglipVisionTower):
    """
    加载指定路径的图像，使用 vision_processor 进行预处理，然后
    利用 SiglipVisionTower 模型提取特征。

    参数：
        image_path: 图像的文件路径
        model: 已实例化的 SiglipVisionTower 对象

    返回：
        tensor 格式的视觉特征，其 shape 通常为 (B, T, W, H, hidden_size)
    """
    # 打开图像并转换为 RGB
    img = Image.open(image_path).convert("RGB")
    # 使用 vision_processor 进行预处理，返回 tensor 格式的 pixel_values
    inputs = model.vision_processor(img, return_tensors="pt")
    pixel_values = inputs["pixel_values"]  # shape (B, C, H, W)
    # 得到模型输出的视觉特征
    features = model(pixel_values)
    return features

def send_features_to_llm(features):
    """
    示例函数：如何将提取的视觉特征传入其他开源 LLM（例如基于 langchain 的 qwen / deepseek）。
    
    说明：
      1. 目前大部分 LLM 主要处理文本输入，因此你可能需要设计一个映射模块，
         如线性投影，将视觉特征转换为文本提示（prompt）或直接作为上下文输入。
      2. 此处给出了简单示例，将特征进行 flatten 后转换成字符串（仅作示意，实际应用中不建议直接传输大量数值）。
    """
    # 将特征 flatten 成一维数组，并转为 list（注意：此转换可能非常庞大，实际应用中需要合适的维度投影）
    features_flat = features.flatten().detach().cpu().numpy().tolist()
    # 仅截取部分数值进行展示
    prompt = f"Extracted visual features: {features_flat[:10]} ... (truncated)"
    
    # 示例：如何利用 langchain 等接口进行进一步调用（以下代码仅为伪代码）
    # from langchain.llms import OpenAI
    # llm = OpenAI(model_name="qwen", temperature=0.7)
    # response = llm(prompt)
    # print("LLM response:", response)

    print("LLM prompt example:")
    print(prompt)

# -----------------------
# 主函数：示例如何使用 SiglipVisionTower 提取特征并显示
# -----------------------

if __name__ == "__main__":
    # 示例预训练模型路径，这里需要替换为实际可用的 Siglip 模型
    model_name_or_path = "siglip-base"  # 替换为实际模型名称或路径

    # 构造一个简单的 VisionTowerConfig 作为模型配置
    config = VisionTowerConfig(
        vision_tower_name=model_name_or_path,
        img_size=224,       # 输入图像的尺寸
        patch_size=16,      # patch 尺寸
        hidden_size=768,    # 特征维度
        num_frames=1        # 单帧
    )

    # 初始化 SiglipVisionTower 模型
    model = SiglipVisionTower(model_name_or_path, config)

    # 指定要提取特征的图像文件路径（请确保该文件存在）
    image_path = "example.jpg"  # 替换为实际图像路径
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"请确保示例图像文件存在: {image_path}")

    # 提取图像视觉特征
    features = extract_features_from_image(image_path, model)
    print("Extracted features shape:", features.shape)

    # 使用 matplotlib 在画布中展示特征图（此处将 hidden 通道取平均后显示热力图）
    try:
        import matplotlib.pyplot as plt
        # 假设 features shape 为 (B, T, W, H, hidden_size)，取第一个样本第一帧
        feature_map = features[0, 0].mean(dim=-1).detach().cpu().numpy()  # 平均 hidden_size 维度
        plt.figure(figsize=(6, 6))
        plt.imshow(feature_map, cmap="viridis")
        plt.title("Visual Feature Map")
        plt.colorbar()
        plt.show()
    except ImportError:
        print("matplotlib 未安装，跳过特征图可视化。")

    # 演示如何将特征传入其他 LLM
    send_features_to_llm(features) 