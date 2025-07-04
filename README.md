# 🧠 LeoNet: Brain-Inspired Transformer for Typed Language-to-Motor Action

This repository contains a demonstration of **LeoNet**, a biologically inspired transformer architecture that unifies **language understanding** with **motor action execution**.

LeoNet transforms **typed natural language commands** (e.g., "move left", "scroll down") into **motor control vectors** capable of moving the mouse pointer across the screen.

---

## 📄 Files Included

| Filename                    | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `Leonet_model.py`           | The LeoNet transformer architecture definition.                            |
| `leonet_command_vision_500.jsonl` | A small sample dataset of command-to-motor pairs (8x128 motion vectors).  |
| `leonet_train_pipeline.py`  | The model training script (text + motor prediction).                        |
| `Leonet_inference.py`       | Inference script for typing a command and observing real mouse movement.    |
| `motor_executor.py`         | Utility to translate motor vectors into real mouse motion using PyAutoGUI. |

---

## 📚 Introduction (From Research Paper)

LeoNet introduces a novel architecture inspired by the brain’s ability to convert **cognition into action**. The model accepts language commands and outputs two parallel predictions:

- **Language stream**: Predicts the next token in the sequence (self-supervised loss).
- **Motor stream**: Outputs a 1024-dim vector reshaped as 8x128 representing a structured motor action (cursor movement).

The motor stream is trained to **directly map commands like "move right" into cursor deltas** like `[+200, 0]`, simulating brain-to-muscle control.

This prototype bridges the gap between **natural language processing and motor control**, setting the foundation for fully embodied LLM-based agents.

---

## ⚙️ Setup Instructions

### 🔧 Installation

Make sure you have Python 3.9+ and install dependencies:

```bash
pip install torch pyautogui pyttsx3
```

### 📦 Files Needed

Ensure the following files are in the same folder:

- `Leonet_model.py`
- `leonet_command_vision_500.jsonl`
- `leonet_train_pipeline.py`
- `Leonet_inference.py`

---

## 🏋️‍♀️ Train the Model 

You can train the LeoNet model on the dataset using:

```bash
python leonet_train_pipeline.py
```

After training, this will create:

```bash
leonet_demo.pth
```

---

## 🚀 Run the Demo (Type a Command)

```bash
python Leonet_inference.py
```

You’ll see:

```
⌨️ Type a command to move (e.g., 'move left', 'scroll down')
🛑 Type 'quit' to exit.
Command:
```

Type:
```text
move left
```

LeoNet will:
- Predict motor delta like `dx = -200`, `dy = 0`
- Move your mouse accordingly
-  Speak back "Movement done"

---

## 🧠 LeoNet Architecture Summary

LeoNet’s transformer has:
- **Cognitive head**: Self-supervised text modeling
- **Motor head**: 1024-dim continuous vector reshaped to (8, 128)

Motor output is trained using MSE loss to match known action deltas from data like:

```json
{
  "input_ids": [13, 14, 17, 4, 26, 26, 26, 26],
  "target_ids": [14, 17, 4, 26, 26, 26, 26, 0],
  "motor_output": [[-200.0, 0.0], [0.0, 0.0], ..., [0.0, 0.0]]
}
```

## 🔍 LeoNet vs Other Transformer Models

| Model         | Parameters (M) | FLOPs (GFLOPs) | Inference Latency (ms) | Motor Output Support | Dual Output (Lang + Motor) | Use Case Suitability     |
|---------------|----------------|----------------|--------------------------|----------------------|-----------------------------|--------------------------|
| **LeoNet**    | **29.55**      | **0.818**      | **54.5**                 | ✅ 1024-dim          | ✅ Yes                      | Real-world dual-task     |
| TinyBERT      | 14.5           | 1.3            | ~66                      | ❌ None              | ❌ No                       | Lightweight NLP          |
| DistilBERT    | 66             | 3.8            | ~115                     | ❌ None              | ❌ No                       | Faster BERT              |
| BERT Base     | 110            | 12.0           | ~150                     | ❌ None              | ❌ No                       | Deep language tasks      |
| GPT-2 Small   | 124            | 15.5           | ~180                     | ❌ None              | ❌ No                       | Generative tasks         |




## 🤖 Deploy LeoNet on Raspberry Pi / Jetson (Edge Robotics Ready)

LeoNet is designed not just for research, but for **real-world deployment** in low-resource environments like **Raspberry Pi** and **NVIDIA Jetson** platforms.

### ✅ Why LeoNet is Edge-Ready

| Feature                             | Supported |
|-------------------------------------|-----------|
| Dual Output (Language + Motor)      | ✅ Yes     |
| Low FLOPs (~0.818 GFLOPs)           | ✅ Yes     |
| Compact Model (~29.5M Parameters)   | ✅ Yes     |
| Inference Latency (~54 ms)          | ✅ Yes     |
| Suitable for Real-Time Robotics     | ✅ Yes     |
| Runs on Jetson Nano / Xavier / Pi 4 | ✅ Yes     |

---

### 🧠 Use Cases

- 🕹️ Voice-controlled robots and actuators  
- 🖱️ Mouse and UI agents controlled by language  
- 🧠 Cognitive agents with both interpretation and action  
- 🤖 Real-time robots using natural language as control

---

### 🔧 Deployment Tips

- ✅ Convert model to **TorchScript** or **ONNX** for optimized runtime
- ✅ Use **int8 quantization** to reduce model size for edge hardware
- ✅ Batch size = 1 for lowest latency on CPU or GPU
- ✅ Works on:
  - Raspberry Pi 4 (ARM64, with PyTorch Lite)
  - Jetson Nano / Xavier NX (JetPack + TensorRT)
  - Any Linux-based embedded board with Python3 + PyTorch

---

### 📦 Example Application

> **Input**: `"move cursor right"`  
> **LeoNet Output**: `[0.12, 0.00, ..., -0.04]`  
> **Action**: Simulated or real motor movement (e.g., servo, cursor, wheel)

---

LeoNet bridges **language understanding and physical action**, making it ideal for embodied agents, GUI automation, and embedded AI.




---

## 📈 Citation (for Research)

```
Silpeshkumar Jitendrabhai Patel. 
LeoNet: A Brain-Inspired Transformer for Dual Cognitive and Motor Output in Real-World Environments. 
TechRxiv Preprint, 2025.
```

---

## 🔒 License

This project is released under the **CC BY 4.0 License**. You are free to use, modify, and distribute with attribution.

---

## 🙌 Acknowledgements

This project demonstrates a proof of concept for building **LLM-based motion agents** that "think and act" in real-world environments. It combines:

- NLP Transformers
- Motor simulation
- Dataset generation
- Self-supervised learning
