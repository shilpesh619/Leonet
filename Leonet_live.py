import whisper
import torch
import pyautogui
import time
import sounddevice as sd
import numpy as np
from PIL import ImageGrab
from leonet_model_vision_v2_dropout import LeoNetVisionV2
from command_dataset_vision import preprocess_image_tensor

# Load Whisper
whisper_model = whisper.load_model("base")  # You can try "small" or "medium" if GPU is strong

# Load LeoNet Vision model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeoNetVisionV2(vocab_size=27).to(device)
model.load_state_dict(torch.load("leonet_v2_vision.pth", map_location=device))
model.eval()

# Tokenizer
vocab = list("abcdefghijklmnopqrstuvwxyz ")
def tokenize(text):
    return [vocab.index(c) if c in vocab else 26 for c in text.lower()][:8] + [0] * max(0, 8 - len(text))

# Valid commands
valid_commands = [
    "move right", "move left", "move up", "move down",
    "click center", "scroll down", "scroll up", "quit"
]

def record_audio(duration=3, samplerate=16000):
    print("🎤 Listening...")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

print("🎤 LeoNet Whisper Agent Running (say 'quit' to exit)...")

while True:
    audio_data = record_audio()
    print("🧠 Transcribing...")
    result = whisper_model.transcribe(audio_data, language="en", fp16=False)
    text = result['text'].lower().strip()
    print(f"🗣️ Command heard: '{text}'")

    if not text:
        continue

    if "quit" in text:
        print("👋 Exiting agent.")
        break

    if any(text.startswith(cmd) for cmd in valid_commands):
        token_ids = tokenize(text)
        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

        screenshot = ImageGrab.grab()
        image_tensor = preprocess_image_tensor(screenshot).unsqueeze(0).to(device)

        with torch.no_grad():
            _, motor_output = model(input_tensor, image_tensor)
            vec = motor_output[0][0].cpu().numpy()
            vec = vec / 2 + 0.5  # scale back to [0, 1]

        x = int(vec[0] * pyautogui.size().width)
        y = int(vec[1] * pyautogui.size().height)
        print(f"🖱️ Moving cursor to ({x}, {y})")
        pyautogui.moveTo(x, y, duration=0.5)

    elif text.startswith("type "):
        content = text.replace("type ", "", 1)
        print(f"⌨️ Typing: {content}")
        pyautogui.typewrite(content)

    elif text.startswith("search "):
        query = text.replace("search ", "", 1)
        print(f"🔍 Searching: {query}")
        pyautogui.typewrite(query)
        time.sleep(0.2)
        pyautogui.press("enter")

    elif "press enter" in text:
        print("⏎ Pressing Enter")
        pyautogui.press("enter")

    elif "right click" in text or "right-click" in text:
        print("🖱️ Right-clicking")
        pyautogui.click(button="right")

    else:
        print(f"❌ Ignored: Not a valid command → '{text}'")
