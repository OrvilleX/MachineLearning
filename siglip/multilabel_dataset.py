from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import numpy as np

class MultiLabelDataset(Dataset):
  def __init__(self, root, df, transform):
    self.root = root
    self.df = df
    self.transform = transform

  def __getitem__(self, idx):
    item = self.df.iloc[idx]
    # get image
    image_path = os.path.join(self.root, item["Image_Name"])

    # 处理可能缺失的图像文件
    if not os.path.exists(image_path):
        return None  # 在collate_fn中会处理这些None值

    image = Image.open(image_path).convert("RGB")

    # 图像预处理流程（与模型预期输入一致）
    pixel_values = self.transform(image)  # 包含归一化等操作

    # 多标签处理（多个独立二分类问题）
    labels = item[2:].values.astype(np.float32)  # 从DataFrame提取多标签
    labels = torch.from_numpy(labels)  # 转换为张量

    return pixel_values, labels

  def __len__(self):
    return len(self.df)