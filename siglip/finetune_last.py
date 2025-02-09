import pandas as pd
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from multilabel_dataset import MultiLabelDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

# 自定义的collate函数，处理可能存在的空数据
def collate_fn(batch):
    # 过滤掉 None（当图像文件不存在时返回的None）
    batch = [item for item in batch if item is not None]
    # 处理全空批次的情况
    if len(batch) == 0:
        return None
    # 堆栈处理后的有效数据
    data = torch.stack([item[0] for item in batch])
    target = torch.stack([item[1] for item in batch])
    return data, target

class AverageMeter(object):
    """用于跟踪并计算指标（如损失、准确率）的移动平均值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 总和
        self.count = 0  # 计数

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n  # 累加值（考虑批次大小）
        self.count += n
        self.avg = self.sum / self.count  # 计算新的平均值

# 模型和数据处理配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "/root/autodl-tmp/siglip-so400m-patch14-384"  # 使用预训练的SigLIP模型
df = pd.read_csv("/root/autodl-tmp/multilabel_modified/multilabel_classification(2).csv")

labels = list(df.columns)[2:]
id2label = {id: label for id, label in enumerate(labels)}

processor = AutoImageProcessor.from_pretrained(model_id, device=device)
model = AutoModelForImageClassification.from_pretrained(model_id, problem_type="multi_label_classification", id2label=id2label)
model = model.to(device)

size = processor.size["height"]
mean = processor.image_mean
std = processor.image_std

# 数据预处理流程（需与模型预训练时的处理一致）
transform = Compose([
    Resize((size, size)),  # 调整到模型预期尺寸
    ToTensor(),  # 转换为张量
    Normalize(mean=mean, std=std),  # 使用模型特定的归一化参数
])

train_dataset = MultiLabelDataset(root="/root/autodl-tmp/multilabel_modified/images",
                                  df=df, transform=transform)

# 创建数据加载器时指定自定义的collate函数
train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=2, shuffle=True)

# 训练循环关键部分
optimizer = AdamW(model.parameters(), lr=5e-5)  # 使用适合Transformer的优化器
losses = AverageMeter()  # 跟踪训练损失

model.train()
for epoch in range(10):
    for idx, batch in enumerate(tqdm(train_dataloader)):
        # 跳过无效批次（当整个批次都是无效数据时）
        if batch is None:
            continue
        
        pixel_values, labels = batch

        optimizer.zero_grad()

        # 混合精度训练（虽然代码中没有显式使用，但模型可能自动处理）
        outputs = model(
            pixel_values=pixel_values.to(device),
            labels=labels.to(device),  # 多标签分类损失会自动计算
        )
        
        # 梯度累积和参数更新
        loss = outputs.loss
        losses.update(loss.item(), pixel_values.size(0))
        loss.backward()
        optimizer.step()

        if idx % 2000 == 0:
            print('Epoch: [{0}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, loss=losses,))

model.save_pretrained("./saved_model/")
