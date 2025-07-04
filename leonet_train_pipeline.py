# leonet_train_pipeline.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import os
from Leonet_model import LeoNet  # Your model file

# ---- 1. Dataset Loader ----
class DemoCommandDataset(Dataset):
    def __init__(self, jsonl_path="leonet_command_vision_500.jsonl"):
        assert os.path.exists(jsonl_path), f"Dataset not found: {jsonl_path}"
        self.samples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        target_ids = torch.tensor(sample["target_ids"], dtype=torch.long)
        motor_vec = sample["motor_output"]

        # ✅ Cleaned & corrected motor reshaping
        if isinstance(motor_vec[0], list) and len(motor_vec) == 8 and len(motor_vec[0]) == 128:
            motor_tensor = torch.tensor(motor_vec, dtype=torch.float32)
        elif isinstance(motor_vec, list) and len(motor_vec) == 1024:
            motor_tensor = torch.tensor(motor_vec, dtype=torch.float32).view(8, 128)
        else:
            flat = motor_vec if isinstance(motor_vec, list) else []
            padded = (flat + [0.0] * 1024)[:1024]
            motor_tensor = torch.tensor(padded, dtype=torch.float32).view(8, 128)

        return input_ids, target_ids, motor_tensor

# ---- 2. Collate Function ----
def collate_fn(batch):
    input_ids, target_ids, motor_outputs = zip(*batch)
    return (
        torch.stack(input_ids),
        torch.stack(target_ids),
        torch.stack(motor_outputs)
    )

# ---- 3. Training Loop ----
def train_leonet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeoNet(vocab_size=27).to(device)

    dataset = DemoCommandDataset("leonet_command_vision_500.jsonl")
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    criterion_text = nn.CrossEntropyLoss()
    criterion_motor = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(3):
        model.train()
        total_loss = 0.0
        for input_ids, target_ids, motor_targets in loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            motor_targets = motor_targets.to(device)

            logits, motor_output = model(input_ids)
            loss_text = criterion_text(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss_motor = criterion_motor(motor_output, motor_targets.view(motor_targets.size(0), -1))
            loss = loss_text + loss_motor

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")

    return model  # ✅ Return the trained model

# ---- Save Model ----
if __name__ == "__main__":
    trained_model = train_leonet()
    torch.save(trained_model.state_dict(), "leonet_demo.pth")
    print("✅ Saved as leonet_demo.pth")
