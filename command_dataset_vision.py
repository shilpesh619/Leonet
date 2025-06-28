# command_dataset_vision.py
import os, json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CommandVisionDataset(Dataset):
    def __init__(self, path="leonet_command_vision.jsonl", image_folder="screenshots"):
        self.samples = []
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                item["image_path"] = os.path.join(image_folder, item["image_name"])
                self.samples.append(item)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        target_ids = torch.tensor(item["target_ids"], dtype=torch.long)
        motor_output = torch.tensor(item["motor_output"], dtype=torch.float32).view(8, 128)
        image = Image.open(item["image_path"]).convert("RGB")
        image_tensor = self.transform(image)
        return input_ids, target_ids, motor_output, image_tensor

# ✅ Add this for live screenshot compatibility
def preprocess_image_tensor(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    if isinstance(img, Image.Image):
        return transform(img)
    else:
        return transform(Image.open(img).convert("RGB"))
