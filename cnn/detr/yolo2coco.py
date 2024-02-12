import json
import os
from PIL import Image

# 图像和标注目录
img_dir = 'path/to/images'
annot_dir = 'path/to/annotations'
output_file = 'metadata.jsonl'

# 类别名称列表，根据实际情况调整
class_names = ['class1', 'class2', 'class3']

# 创建 metadata.jsonl 文件
with open(output_file, 'w') as f:
    for img_name in os.listdir(img_dir):
        if not img_name.endswith('.jpg'):
            continue

        # 获取图像尺寸
        img_path = os.path.join(img_dir, img_name)
        with Image.open(img_path) as img:
            width, height = img.size

        # 解析对应的 YOLO 标注文件
        annot_path = os.path.join(annot_dir, img_name.replace('.jpg', '.txt'))
        annotations = []
        with open(annot_path, 'r') as annot_file:
            for line in annot_file:
                class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())
                # 转换为 [x_min, y_min, x_max, y_max] 格式
                x_min = (x_center - bbox_width / 2) * width
                y_min = (y_center - bbox_height / 2) * height
                x_max = (x_center + bbox_width / 2) * width
                y_max = (y_center + bbox_height / 2) * height
                annotations.append({
                    'bbox': [x_min, y_min, x_max, y_max],
                    'category': class_names[int(class_id)]
                })

        # 创建 JSON 对象
        json_obj = {
            'image': img_name,
            'width': width,
            'height': height,
            'annotations': annotations
        }

        # 写入文件
        f.write(json.dumps(json_obj) + '\n')
