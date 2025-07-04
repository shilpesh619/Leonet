import pyautogui
import time

def execute_motor_command(motor_tensor, vocab=None, verbose=True):
    """
    Takes the motor output tensor of shape [8, 128] and executes the action.

    - motor_tensor: Tensor or list with shape [8, 128]
    - vocab: (optional) for logging, not needed to execute
    - verbose: Whether to print debug output
    """

    # Extract the first vector (assumed to hold primary motion info)
    if hasattr(motor_tensor, 'detach'):
        motor_vector = motor_tensor[0].detach().cpu().numpy()
    else:
        motor_vector = motor_tensor[0]  # assume already NumPy-like

    dx = float(motor_vector[0])
    dy = float(motor_vector[1])
    click_type = int(round(motor_vector[2])) if len(motor_vector) > 2 else 0

    if verbose:
        print(f"🖱️ Motor Action → dx: {dx:.2f}, dy: {dy:.2f}, click: {click_type}")

    # Perform movement
    pyautogui.moveRel(dx, dy, duration=0.2)

    # Optional click logic
    if click_type == 1:
        pyautogui.click()
    elif click_type == 2:
        pyautogui.rightClick()
    elif click_type == 3:
        pyautogui.doubleClick()

    # Add more click modes if your model supports them

