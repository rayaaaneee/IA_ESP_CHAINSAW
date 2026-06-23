import os
import sys

from invoke import task

# Active virtual environment is assumed when this script runs
IS_WIN = sys.platform == "win32"
PYTHON = os.path.join("venv", "Scripts", "python.exe") if IS_WIN else os.path.join("venv", "bin", "python")
PIP = os.path.join("venv", "Scripts", "pip.exe") if IS_WIN else os.path.join("venv", "bin", "pip")
PIO = os.path.join("venv", "Scripts", "pio.exe") if IS_WIN else os.path.join("venv", "bin", "pio")

@task
def train(c):
    # Extract features and train the AI model.
    print("Extracting audio features...")
    c.run(f'"{PYTHON}" training/extract_features.py')
    print("Training TensorFlow model...")
    c.run(f'"{PYTHON}" training/train.py')

@task
def convert(c):
    # Convert the trained model to TensorFlow Lite format (model.h).
    print("Converting model to TFLite header...")
    c.run(f'"{PYTHON}" training/convert_to_tflite.py')

@task(pre=[convert])
def build(c):
    # Compile the C++ firmware for the ESP32 using PlatformIO.
    print("Building ESP32 firmware...")
    with c.cd("firmware"):
        c.run(f'"{PIO}" run')

@task
def upload(c):
    # Upload the compiled firmware to the connected ESP32 board.
    print("Flashing firmware to ESP32...")
    with c.cd("firmware"):
        c.run(f'"{PIO}" run --target upload')

@task
def monitor(c):
    # Open the PlatformIO serial device monitor for debugging.
    print("Opening serial monitor...")
    with c.cd("firmware"):
        c.run(f'"{PIO}" device monitor')