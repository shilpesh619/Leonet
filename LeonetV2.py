
import torch
from leonet_model_vision_v2_dropout import LeoNetVisionV2
from command_dataset_vision import CommandVisionDataset
import pyautogui
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeoNetVisionV2(vocab_size=27).to(device)
model.load_state_dict(torch.load("leonet_v2_vision.pth", map_location=device))
model.eval()

vocab = list("abcdefghijklmnopqrstuvwxyz ")

def tokenize(cmd):
    tokens = [vocab.index(c) for c in cmd if c in vocab]
    return tokens[:8] + [0] * max(0, 8 - len(tokens))

# Load a screenshot for simulation
ds = CommandVisionDataset(path="leonet_command_vision_500.jsonl", image_folder="screenshots")
_, _, _, img = ds[0]
image_tensor = img.unsqueeze(0).to(device)

while True:
    cmd = input("\n💬 Type a command ('q' to quit): ").lower().strip()
    if cmd == "q":
        break

    tokens = tokenize(cmd)
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        _, motor_output = model(input_tensor, image_tensor)
        vec = motor_output[0][0].cpu().numpy()  # Use first motor vector

        # Denormalize back to [0, 1] from [-1, 1]
        vec = vec / 2 + 0.5
        x = int(vec[0] * pyautogui.size().width)
        y = int(vec[1] * pyautogui.size().height)

        print(f"🖱️ Simulated move to ({x}, {y})")
        pyautogui.moveTo(x, y, duration=0.5)
