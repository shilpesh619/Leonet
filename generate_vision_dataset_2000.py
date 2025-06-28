
import os
import json
import random
import pyautogui
import time
from PIL import ImageGrab

COMMANDS = [
    "move right", "move left", "move up", "move down",
    "click center", "click left", "click right",
    "scroll up", "scroll down", "scroll slowly",
    "type hello", "type a", "type z"
]

# Create folders
os.makedirs("screenshots", exist_ok=True)

with open("leonet_command_vision_2000.jsonl", "w") as f:
    for i in range(2000):
        cmd = random.choice(COMMANDS)

        # Random X, Y target in normalized screen space
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        click = 1.0 if "click" in cmd else 0.0
        scroll = 1.0 if "scroll" in cmd else 0.0

        motor_vector = [x, y, click, scroll, 0.0]

        # Capture screen (pause briefly to allow variation)
        time.sleep(0.05)
        img = ImageGrab.grab()
        img_path = os.path.join("screenshots", f"img_{i}.png")
        img.save(img_path)

        entry = {
            "input_ids": [ord(c)-97 if c.isalpha() else 26 for c in cmd][:8] + [0]*(8 - len(cmd)),
            "target_ids": [ord(c)-97 if c.isalpha() else 26 for c in cmd][:8] + [0]*(8 - len(cmd)),
            "motor_output": motor_vector,
            "image_path": f"img_{i}.png"
        }

        f.write(json.dumps(entry) + "\n")
        print(f"✅ Sample {i}: {cmd} => {motor_vector}")
