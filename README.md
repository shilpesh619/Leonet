# Leonet
# LeoNet: A Biologically-Inspired Transformer with Expansion–Contraction Dynamics and Dual Cognitive–Motor Outputs

![LeoNet Architecture](![Leonet](https://github.com/user-attachments/assets/f2e8f85b-bb16-48c0-9bd2-6a1e9a1024aa)
)
)

## 🧠 Overview

**LeoNet** is a novel transformer architecture inspired by biological neurons, designed to generate both:
- **Cognitive outputs** (natural language, internal reasoning)
- **Motor outputs** (mouse, keyboard, or robotic actions)

It introduces a custom **Expansion–Contraction Block (ECB)** that splits the hidden representation during expansion, generating parallel pathways:
- One for language generation (cognitive)
- One for real-time control (motor)

> LeoNet can power embodied AI agents, desktop assistants, or robots that think and act in real time.

---

## 🔧 Architecture Components

- **Multi-Head Attention** (standard)
- **Expansion–Contraction Block**
  - Linear Expand (→ 4× hidden dim)
  - Clone to:
    - **Motor Output Path**
    - **Cognitive Path**
  - Contract merged output back to hidden dim
- **Dual Output Heads**:
  - Softmax for text
  - Linear for motor vector

### ⛓️ ECB Flow

Input x ∈ ℝᵈ
↓
x_e = Linear_expand(x) ∈ ℝ⁴ᵈ
↓
x_m = Linear_motor(x_e) # Motor stream
x_c = Linear_cognitive(x_e) # Cognitive stream
↓
x_out = Linear_contract([x_c; x_m]) ∈ ℝᵈ



---

## 🤖 Applications

- 🖱️ **Desktop Automation**  
  Motor output feeds mouse/keyboard actions (e.g., via PyAutoGUI)

- 🦿 **Humanoid Robotics**  
  Output can control arms, head, or camera focus

- 🧩 **Cognitive + Motor Agents**  
  LeoNet separates planning vs execution — like the brain

---

## 📦 Project Structure

```bash
leonet/
├── model/
│   ├── leonet.py               # Architecture code
│   ├── layers.py               # ECB and transformer blocks
├── training/
│   └── train.py                # Example training loop
├── configs/
│   └── leonet_architecture.json
├── images/
│   └── leonet_diagram.png      # Architecture diagram
├── README.md
└── leonet_paper.tex            # ArXiv-ready paper


Citation

@misc{patel2025leonet,
  author = {Silpeshkumar J. Patel},
  title = {LeoNet: A Biologically-Inspired Transformer with Expansion--Contraction Dynamics and Dual Cognitive--Motor Outputs},
  year = {2025},
  url = {https://github.com/yourusername/leonet}
}
