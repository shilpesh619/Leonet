# Leonet
# LeoNet: A Biologically-Inspired Transformer with Expansion–Contraction Dynamics and Dual Cognitive–Motor Outputs

# 🧠 LeoNet-Vision Agent

LeoNet is a biologically inspired transformer-based AI that integrates:
- 🗣️ Spoken natural language commands
- 👁️ Visual desktop context (screenshots)
- 🤖 Motor output to control the mouse and keyboard

## Features
- Real-time voice command execution
- Offline screen-based cursor control
- Dual-stream transformer: text + motor output
- Custom dataset with 8x128 motor vector encoding

## Architecture
LeoNet uses a transformer backbone that forks into:
1. A language modeling head
2. A motor vector predictor (8x128 output)

It receives both tokenized command text and a screenshot tensor.

## Setup
```bash
pip install -r requirements.txt


Citation

@misc{patel2025leonet,
  author = {Silpeshkumar J. Patel},
  title = {LeoNet: A Biologically-Inspired Transformer with Expansion--Contraction Dynamics and Dual Cognitive--Motor Outputs},
  year = {2025},
  url = {https://github.com/shilpesh619/leonet}
}
