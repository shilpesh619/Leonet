# leonet_train_pipeline.py (Fixed motor output length handling + live command test)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import os
from leonet_model import LeoNet

# ----- Auto-generate dataset if not exists -----
def generate_command_dataset(path="leonet_command_dataset.jsonl"):
    import random
    COMMAND_TO_VECTOR = {
        "move right and click": [0.8, 0.5, 0.95, 0.5, 0.0],
        "move left and click": [0.2, 0.5, 0.95, 0.5, 0.0],
        "scroll down": [0.5, 0.5, 0.0, 0.2, 0.0],
        "scroll up": [0.5, 0.5, 0.0, 0.8, 0.0],
        "type a": [0.5, 0.5, 0.0, 0.5, 0.0],
        "type z": [0.5, 0.5, 0.0, 0.5, 1.0],
        "move up": [0.5, 0.2, 0.0, 0.5, 0.0],
        "move down": [0.5, 0.8, 0.0, 0.5, 0.0],
        "click once": [0.5, 0.5, 0.95, 0.5, 0.0],
        "move and scroll": [0.7, 0.7, 0.0, 0.6, 0.0]
    }
    vocab = list("abcdefghijklmnopqrstuvwxyz ")
    dataset = []
    for _ in range(100):
        command, vector = random.choice(list(COMMAND_TO_VECTOR.items()))
        tokens = [vocab.index(c) for c in command if c in vocab]
        token_ids = tokens[:8] + [0] * max(0, 8 - len(tokens[:8]))

        vec_min, vec_max = min(vector), max(vector)
        normalized_vector = [(v - vec_min) / (vec_max - vec_min + 1e-6) for v in vector]
        repeat_count = (1024 + len(normalized_vector) - 1) // len(normalized_vector)
        motor_vec = (normalized_vector * repeat_count)[:1024]
        if len(motor_vec) < 1024:
            motor_vec += [0.0] * (1024 - len(motor_vec))

        entry = {
            "input_ids": token_ids,
            "target_ids": token_ids,
            "motor_output": motor_vec
        }
        dataset.append(entry)
    with open(path, "w") as f:
        for item in dataset:
            json.dump(item, f)
            f.write("\n")
    print("✅ Auto-generated leonet_command_dataset.jsonl with guaranteed shape 8x128")

# ----- 1. Command Dataset Loader -----
class CommandDataset(Dataset):
    def __init__(self, path="leonet_command_dataset.jsonl"):
        if not os.path.exists(path):
            generate_command_dataset(path)
        self.samples = []
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        input_ids = torch.tensor(item["input_ids"][:8], dtype=torch.long)
        target_ids = torch.tensor(item["target_ids"][:8], dtype=torch.long)
        motor_output = torch.tensor(item["motor_output"], dtype=torch.float32).view(8, 128)
        return input_ids, target_ids, motor_output

# ----- 2. Collate Function -----
def collate_fn(batch):
    input_ids, target_ids, motor_targets = zip(*batch)
    input_ids = torch.stack(input_ids)
    target_ids = torch.stack(target_ids)
    motor_targets = torch.stack(motor_targets)
    return input_ids, target_ids, motor_targets

# ----- 3. Training + Motor Output Test -----
def train_leonet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 27
    model = LeoNet(vocab_size=vocab_size).to(device)
    dataset = CommandDataset()
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    criterion_text = nn.CrossEntropyLoss()
    criterion_motor = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch in loader:
            input_ids, target_ids, motor_targets = [x.to(device) for x in batch]
            logits, motor_outputs = model(input_ids)

            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = target_ids.view(-1)
            loss_text = criterion_text(logits_flat, targets_flat)

            loss_motor = sum(criterion_motor(mo, motor_targets) for mo in motor_outputs) / len(motor_outputs)
            loss = loss_text + loss_motor

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")

    print("\nTesting motor output on desktop...")
    model.eval()
    with torch.no_grad():
        sample_input = torch.randint(0, vocab_size, (1, 8)).to(device)
        logits, motor_outputs = model(sample_input)

        from motor_executor import execute_motor_command
        vocab = list("abcdefghijklmnopqrstuvwxyz ")
        execute_motor_command(motor_outputs[-1][0, 0], vocab)

    torch.save(model.state_dict(), "leonet_model.pth")

# ----- 4. Live Command Inference -----
def test_live_command():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = list("abcdefghijklmnopqrstuvwxyz ")
    model = LeoNet(vocab_size=27).to(device)
    model.load_state_dict(torch.load("leonet_model.pth"))
    model.eval()

    while True:
        cmd = input("\n💬 Type a command ('q' to quit): ").lower().strip()
        if cmd == "q":
            break

        token_ids = [vocab.index(c) for c in cmd if c in vocab]
        token_ids = token_ids[:8] + [0] * max(0, 8 - len(token_ids))
        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            _, motor_outputs = model(input_tensor)
            from motor_executor import execute_motor_command
            execute_motor_command(motor_outputs[-1][0, 0], vocab)

if __name__ == "__main__":
    train_leonet()
    test_live_command()
