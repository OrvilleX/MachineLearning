import huggingface_hub
from huggingface_hub import notebook_login
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForObjectDetection, TrainingArguments, Trainer
import numpy as np
import torch
import albumentations

huggingface_hub.login()
checkpoint = "microsoft/conditional-detr-resnet-50"
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
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])

        area.append(objects["area"])
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
    cppe5 = load_dataset("cppe-5")
    categories = cppe5["train"].features["objects"].feature["category"].names
    id2label = {index: x for index, x in enumerate(categories, start=0)}
    label2id = {v: k for k, v in id2label.items()}
    remove_idx = [590, 821, 822, 875, 876, 878, 879]
    keep = [i for i in range(len(cppe5["train"])) if i not in remove_idx]
    cppe5["train"] = cppe5["train"].select(keep)
    cppe5["train"] = cppe5["train"].with_transform(transform_aug_ann)
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    training_args = TrainingArguments(
        output_dir="con-detr-resnet-50_finetuned_cppe5",
        per_device_train_batch_size=8,
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
        train_dataset=cppe5["train"],
        tokenizer=image_processor,
    )

    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    train()