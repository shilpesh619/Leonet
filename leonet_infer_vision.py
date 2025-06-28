# leonet_infer_vision.py
import torch
from leonet_model_vision import LeoNetVision
from torchvision import transforms
from PIL import Image
import pyautogui

def capture_and_predict(cmd):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeoNetVision(vocab_size=27).to(device)
    model.load_state_dict(torch.load("leonet_vision.pth"))
    model.eval()

    vocab = list("abcdefghijklmnopqrstuvwxyz ")
    token_ids = [vocab.index(c) for c in cmd if c in vocab]
    token_ids = token_ids[:8] + [0] * (8 - len(token_ids))
    input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

    img = pyautogui.screenshot()
    img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])(img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        _, motor_out = model(input_tensor, img)

        from motor_executor import execute_motor_command
        execute_motor_command(motor_out[0, 0], vocab)

if __name__ == "__main__":
    while True:
        cmd = input("💬 Type command (q to quit): ")
        if cmd.lower() == "q":
            break
        capture_and_predict(cmd)
...