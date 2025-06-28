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
LeoNetVision/
├── leonet_model_vision_v2_dropout.py       # Model definition (V2 with dropout)
├── command_dataset_vision.py               # Dataset loader and image preprocessor
├── generate_vision_dataset.py              # Dataset generation script
├── leonet_train_vision.py                  # Training script
├── leonet_infer_vision.py                  # CLI-based inference
├── leonet_cursor_preview.py                # Simulated cursor movement preview
├── leonet_live_agent.py                    # Voice + vision live control agent
├── screenshots/                            # Screenshot image folder
├── leonet_command_vision_500.jsonl         # Dataset file
├── model-en/                               # Whisper or Vosk speech model (optional)
├── README.md                               # Project overview
└── requirements.txt                        # Dependencies
pip install -r requirements.txt

# Train
python leonet_train_vision.py

# Preview cursor
python leonet_cursor_preview.py

# Run Live Agent
python leonet_live_agent.py



Citation

@misc{patel2025leonet,
  author = {Silpeshkumar J. Patel},
  title = {LeoNet: A Biologically-Inspired Transformer with Expansion--Contraction Dynamics and Dual Cognitive--Motor Outputs},
  year = {2025},
  url = {https://github.com/shilpesh619/leonet}
}
