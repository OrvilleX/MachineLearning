import huggingface_hub
from huggingface_hub import notebook_login
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForObjectDetection, TrainingArguments, Trainer
import numpy as np
import torch
import albumentations

huggingface_hub.login()
checkpoint = "microsoft/conditional-detr-resnet-50"
checkpoint = "facebook/detr-resnet-50-dc5"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)


def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i])
        }
        annotations.append(new_ann)
    return annotations


def transform_aug_ann(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    transform = albumentations.Compose(
        [
            albumentations.Resize(480, 480),
            albumentations.HorizontalFlip(p=1.0),
            albumentations.RandomBrightnessContrast(p=1.0)
        ],
        bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"])
    )
    for image, objects in zip(examples["image"], examples["annotations"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(image=image, bboxes=[obj['bbox'] for obj in objects], category=[obj['category_id']
                                                                                        for obj in objects])

        area.append([obj['area'] for obj in objects])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")


def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch


def train():
    # cppe5 = load_dataset("whereAlone/vest")
    cppe5 = load_dataset("imagefolder", data_dir="F:\\hszb\\vest")
    categories = ['f_vest', 'b_vest', 'f_scarf', 'b_scarf', 'f_belt', 'b_belt']
    id2label = {index: x for index, x in enumerate(categories, start=0)}
    label2id = {v: k for k, v in id2label.items()}
    cppe5["validation"] = cppe5["validation"].with_transform(transform_aug_ann)
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    training_args = TrainingArguments(
        output_dir="detr-resnet-50_finetuned_cppe5",
        per_device_train_batch_size=2,
        num_train_epochs=20,
        fp16=True,
        save_steps=200,
        logging_steps=50,
        learning_rate=1e-5,
        weight_decay=1e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=cppe5["validation"],
        tokenizer=image_processor,
    )

    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    # 指定每个分割的文件路径
    data_files = {
        "train": "F:\\hszb\\vest\\train\\metadata.jsonl",
        "validation": "F:\\hszb\\vest\\val\\metadata.jsonl"
    }
    dataset = load_dataset("imagefolder", data_dir="F:\\hszb\\vest")
    dataset.push_to_hub("whereAlone/vest")
    # categories = dataset["train"].features["annotations"].feature["category"].names
    # train()
