import json
import os
import torchvision
import torch
import evaluate
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForObjectDetection

checkpoint = "facebook/detr-resnet-50"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, image_processor, ann_file):
        super().__init__(img_folder, ann_file)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target: converting target to DETR format,
        # resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        encoding = self.image_processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return {"pixel_values": pixel_values, "labels": target}


# format annotations the same as for training, no need for data augmentation
def val_formatted_anns(image_id, objects):
    annotations = []
    for i in range(0, len(objects["id"])):
        new_ann = {
            "id": objects["id"][i],
            "category_id": objects["category"][i],
            "iscrowd": 0,
            "image_id": image_id,
            "area": objects["area"][i],
            "bbox": objects["bbox"][i],
        }
        annotations.append(new_ann)

    return annotations


# Save images and annotations into the files torchvision.datasets.CocoDetection expects
def save_cppe5_annotation_file_images(cppe5):
    categories = cppe5["test"].features["objects"].feature["category"].names
    id2label = {index: x for index, x in enumerate(categories, start=0)}
    output_json = {}
    path_output_cppe5 = f"{os.getcwd()}/cppe5/"

    if not os.path.exists(path_output_cppe5):
        os.makedirs(path_output_cppe5)

    path_anno = os.path.join(path_output_cppe5, "cppe5_ann.json")
    categories_json = [{"supercategory": "none", "id": id, "name": id2label[id]} for id in id2label]
    output_json["images"] = []
    output_json["annotations"] = []
    for example in cppe5:
        ann = val_formatted_anns(example["image_id"], example["objects"])
        output_json["images"].append(
            {
                "id": example["image_id"],
                "width": example["image"].width,
                "height": example["image"].height,
                "file_name": f"{example['image_id']}.png",
            }
        )
        output_json["annotations"].extend(ann)
    output_json["categories"] = categories_json

    with open(path_anno, "w") as file:
        json.dump(output_json, file, ensure_ascii=False, indent=4)

    for im, img_id in zip(cppe5["image"], cppe5["image_id"]):
        path_img = os.path.join(path_output_cppe5, f"{img_id}.png")
        im.save(path_img)

    return path_output_cppe5, path_anno


def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch


def eval():
    cppe5 = load_dataset("cppe-5")
    im_processor = AutoImageProcessor.from_pretrained("whereAlone/detr-resnet-50_finetuned_cppe5")
    path_output_cppe5, path_anno = save_cppe5_annotation_file_images(cppe5["test"])
    test_ds_coco_format = CocoDetection(path_output_cppe5, im_processor, path_anno)
    model = AutoModelForObjectDetection.from_pretrained("whereAlone/detr-resnet-50_finetuned_cppe5")
    module = evaluate.load("ybelkada/cocoevaluate", coco=test_ds_coco_format.coco)
    val_dataloader = torch.utils.data.DataLoader(
        test_ds_coco_format, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_dataloader)):
            pixel_values = batch["pixel_values"]
            pixel_mask = batch["pixel_mask"]

            labels = [
                {k: v for k, v in t.items()} for t in batch["labels"]
            ]  # these are in DETR format, resized + normalized

            # forward pass
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
            results = im_processor.post_process(outputs,
                                                orig_target_sizes)  # convert outputs of model to Pascal VOC format (xmin, ymin, xmax, ymax)

            module.add(prediction=results, reference=labels)
            del batch

    results = module.compute()
    print(results)


if __name__ == "__main__":
    eval()