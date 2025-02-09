import torch
from datasets import load_dataset
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

ds = load_dataset("imagefolder", data_dir="F:\\hszb\\vest")
example = ds['validation'][500]
boxes_xywh = torch.tensor([obj['bbox'] for obj in example['annotations']])
boxes_xyxy = box_convert(boxes_xywh, 'xywh', 'xyxy')
categories = ['f_vest', 'b_vest', 'f_scarf', 'b_scarf', 'f_belt', 'b_belt']
labels = [categories[x['category_id']] for x in example['annotations']]
to_pil_image(
    draw_bounding_boxes(
        pil_to_tensor(example['image']),
        boxes_xyxy,
        colors="red",
        labels=labels
    )
).show()
