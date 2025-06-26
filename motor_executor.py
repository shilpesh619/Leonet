# motor_executor.py
# Use motor output from LeoNet to control mouse and keyboard
import pyautogui
import torch
import time

# Assume output shape: (batch_size, seq_len, motor_dim)
# For this example, we'll use only the first item and interpret motor_dim as:
# [move_x, move_y, click, scroll, type_char_index]

def interpret_motor_output(motor_vector, vocab):
    move_x = int((motor_vector[0].item() - 0.5) * 200)  # [-100, +100] range
    move_y = int((motor_vector[1].item() - 0.5) * 200)
    click = motor_vector[2].item() > 0.8
    scroll = int((motor_vector[3].item() - 0.5) * 10)
    char_index = int(motor_vector[4].item() * len(vocab)) % len(vocab)
    char = vocab[char_index]

    return move_x, move_y, click, scroll, char

def execute_motor_command(motor_vector, vocab):
    move_x, move_y, click, scroll, char = interpret_motor_output(motor_vector, vocab)

    pyautogui.moveRel(move_x, move_y, duration=0.25)
    if click:
        pyautogui.click()
    if scroll != 0:
        pyautogui.scroll(scroll)
    if char.isprintable():
        pyautogui.write(char)

# Demo usage (for test only)
if __name__ == "__main__":
    vocab = list("abcdefghijklmnopqrstuvwxyz ")
    dummy_motor_vector = torch.rand(5)  # Simulate output from LeoNet
    print("Simulated motor vector:", dummy_motor_vector)
    execute_motor_command(dummy_motor_vector, vocab)
    print("Motor action executed.")
