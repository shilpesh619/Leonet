import torch
import pyttsx3
import pyautogui
from Leonet_model import LeoNet

# ------------------- Tokenizer -------------------
vocab = list("abcdefghijklmnopqrstuvwxyz ")
def tokenize(text):
    tokens = [vocab.index(c) if c in vocab else 26 for c in text.lower()]
    return tokens[:8] + [0] * max(0, 8 - len(tokens))

# ------------------- Load Model -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeoNet(vocab_size=27).to(device)
model.load_state_dict(torch.load("leonet_demo.pth", map_location=device))
model.eval()

# ------------------- Init TTS -------------------
tts = pyttsx3.init()

# ------------------- Command Execution -------------------
def run_leonet(command):
    print(f"🧠 Command: {command}")
    tokens = tokenize(command)
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        _, motor_output = model(input_tensor)
    dx = int(motor_output[0, 0].item())
    dy = int(motor_output[0, 1].item())
    print(f"🖱️ Move: dx={dx}, dy={dy}")
    pyautogui.moveRel(dx, dy)
    tts.say("Movement done")
    tts.runAndWait()

# ------------------- Main Loop -------------------
print("⌨️ Type a command to move (e.g., 'move left', 'scroll down')")
print("🛑 Type 'quit' to exit.")

while True:
    command = input("Command: ").strip().lower()
    if not command:
        continue
    if command == "quit":
        print("✅ Exiting.")
        break
    run_leonet(command)
