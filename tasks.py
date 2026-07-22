import os
import sys

from invoke import Collection, Program, task
from invoke.exceptions import ParseError

# Active virtual environment is assumed when this script runs
IS_WIN = sys.platform == "win32"

VENV = os.environ.get("VIRTUAL_ENV")
if VENV is None:
    raise RuntimeError("No virtual environment is active.")

PYTHON = os.path.join(VENV, "Scripts", "python.exe") if IS_WIN else os.path.join(VENV, "bin", "python")
PIP = os.path.join(VENV, "Scripts", "pip.exe") if IS_WIN else os.path.join(VENV, "bin", "pip")
PIO = os.path.join(VENV, "Scripts", "pio.exe") if IS_WIN else os.path.join(VENV, "bin", "pio")
INV = os.path.join(VENV, "Scripts", "invoke.exe") if IS_WIN else os.path.join(VENV, "bin", "invoke")

@task(aliases=["l"])
def list(c):
    # List all available tasks.
    c.run(f'"{INV}" --list')

@task(aliases=["t"])
def train(c):
    # Extract features and train the AI model.
    # Only extract features if cache/manifest are missing or inconsistent.
    print("Checking feature cache...")
    res = c.run(f'"{PYTHON}" training/check_cache.py', warn=True)
    if not res.ok:
        print("Extracting audio features...")
        c.run(f'"{PYTHON}" training/extract_features.py')
    else:
        print("Feature cache is valid; skipping extraction.")
    print("Training TensorFlow model...")
    # We already checked (and possibly refreshed) the cache above, so tell train.py not to re-extract.
    c.run(f'"{PYTHON}" training/train.py --no-extract')

@task(aliases=["c"])
def convert(c):
    # Convert the trained model to TensorFlow Lite format (model.h).
    print("Converting model to TFLite header...")
    c.run(f'"{PYTHON}" training/convert_to_tflite.py')

@task(pre=[convert], aliases=["b"])
def build(c):
    # Compile the C++ firmware for the ESP32 using PlatformIO.
    print("Building ESP32 firmware...")
    with c.cd("firmware"):
        c.run(f'"{PIO}" run')

@task(aliases=["up", "flash", "f"])
def upload(c):
    # Upload the compiled firmware to the connected ESP32 board.
    print("Flashing firmware to ESP32...")
    with c.cd("firmware"):
        c.run(f'"{PIO}" run --target upload')

@task(aliases=["m"])
def monitor(c):
    # Open the PlatformIO serial device monitor for debugging.
    print("Opening serial monitor...")
    with c.cd("firmware"):
        c.run(f'"{PIO}" device monitor')

@task(aliases=["p", "clean"])
def prune(c):
    # Clean up PlatformIO system files to free up space.
    print("Pruning PlatformIO system files...")
    c.run(f'"{PIO}" system prune')

@task(aliases=["d", "deps"])
def dependencies(c):
    # Reset the virtual environment by removing it and reinstalling dependencies.
    print("Compiling dependencies from requirements.in...")
    c.run(f'"{PYTHON}" -m piptools compile --upgrade requirements.in --output-file=requirements.txt')
    
@task(aliases=["s", "sync_deps", "sync", "sd"])
def sync_dependencies(c):
    # Synchronize the virtual environment with the compiled requirements.txt.
    print("Synchronizing virtual environment with requirements.txt...")
    c.run(f'{PYTHON} -m piptools sync requirements.txt')

@task(aliases=["i", "r", "reset", "reinstall"])
def install(c):
    c.run(f'"{PIP}" install --upgrade -r requirements.txt')

@task(aliases=["extract", "e", "ef"])
def extract_features(c):
    # Extract audio features from the dataset and save them to a compressed .npz file.
    print("Extracting audio features to compressed .npz file...")
    c.run(f'"{PYTHON}" training/extract_features.py')
    
@task(aliases=["check_labels", "cl"])
def labels(c):
    # Check the labels in the feature dataset manifest for potential mislabelling.
    print("Checking labels in feature dataset manifest...")
    c.run(f'"{PYTHON}" training/check_labels.py')
    
@task(pre=[extract_features, train, convert], aliases=["fp", "pl", "pipeline"])
def full_pipeline(c):
    # Run the full pipeline: extract features, train the model, and convert to TFLite.
    print("Running full pipeline: extract features, train model, convert to TFLite...")
    pass