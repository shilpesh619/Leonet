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
