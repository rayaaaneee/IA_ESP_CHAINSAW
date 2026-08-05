import os
import sys
from pathlib import Path

from invoke import task

# Active virtual environment is assumed when this script runs
IS_WIN = sys.platform == "win32"

PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING = PROJECT_ROOT / "training"
FIRMWARE = PROJECT_ROOT / "firmware"

VENV = os.environ.get("VIRTUAL_ENV")
if not VENV:
    raise RuntimeError("No active virtual environment detected. Please activate a virtual environment before running tasks.py.")

PYTHON = os.path.join(VENV, "Scripts", "python.exe") if IS_WIN else os.path.join(VENV, "bin", "python")
PIP = os.path.join(VENV, "Scripts", "pip.exe") if IS_WIN else os.path.join(VENV, "bin", "pip")
PIO = os.path.join(VENV, "Scripts", "platformio.exe") if IS_WIN else os.path.join(VENV, "bin", "platformio")
INV = os.path.join(VENV, "Scripts", "invoke.exe") if IS_WIN else os.path.join(VENV, "bin", "invoke")

@task(aliases=["l"])
def list(c):
    # List all available tasks.
    c.run(f'"{INV}" --list')

@task(aliases=["t"])
def train(c):
    # Extract features and train the AI model.
    # Only extract features if cache/manifest are missing or inconsistent.
    with c.cd(TRAINING):
        print("Checking feature cache...")
        res = c.run(f'"{PYTHON}" check.py --cache', warn=True)

        if not res.ok:
            print("Extracting audio features...")
            c.run(f'"{PYTHON}" extract_features.py')
        else:
            print("Feature cache is valid; skipping extraction.")
        print("Training TensorFlow model...")
        # We already checked (and possibly refreshed) the cache above, so tell train.py not to re-extract.
        c.run(f'"{PYTHON}" train.py --no-extract')

@task(aliases=["c", "tflite", "tfl"])
def convert(c):
    # Convert the trained model to TensorFlow Lite format (model.h).
    print("Converting model to TFLite header...")
    with c.cd(TRAINING):
        c.run(f'"{PYTHON}" convert_to_tflite.py')

@task(aliases=["vt", "v-tflite", "v-lite", "vtflite", "v-tfl", "vtfl", "vlite"])
def validate_tflite(c):
    # Validate the generated TensorFlow Lite model by running inference on a sample input.
    with c.cd(TRAINING):
        print("Validating TFLite model with different thresholds, comparing results and generating report...")
        print("\nRunning validation with threshold=0.5...")
        c.run(f'"{PYTHON}" validate_tflite.py')
        print("\nRunning validation with threshold=0.6...")
        c.run(f'"{PYTHON}" validate_tflite.py --threshold=0.6')
        print("\nRunning validation with threshold=0.4...")
        c.run(f'"{PYTHON}" validate_tflite.py --threshold=0.4')
        print("\nRunning validation with threshold=0.3...")
        c.run(f'"{PYTHON}" validate_tflite.py --threshold=0.3')

@task(aliases=["b"])
def build(c):
    # Compile the C++ firmware for the ESP32 using PlatformIO.
    print("Building ESP32 firmware...")
    with c.cd(FIRMWARE):
        c.run(f'"{PIO}" run')

@task(aliases=["up", "u", "flash", "f", "run"])
def upload(c):
    # Upload the compiled firmware to the connected ESP32 board.
    print("Flashing firmware to ESP32...")
    with c.cd(FIRMWARE):
        c.run(f'"{PIO}" run --target upload')

@task(aliases=["m"])
def monitor(c):
    # Open the PlatformIO serial device monitor for debugging.
    print("Opening serial monitor...")
    with c.cd(FIRMWARE):
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
    with c.cd(PROJECT_ROOT):
        c.run(f'"{PYTHON}" -m piptools compile --upgrade requirements.in --output-file=requirements.txt')
    
@task(aliases=["s", "sync_deps", "sync", "sd"], pre=[dependencies])
def sync_dependencies(c):
    # Synchronize the virtual environment with the compiled requirements.txt.
    print("Synchronizing virtual environment with requirements.txt...")
    with c.cd(PROJECT_ROOT):
        c.run(f'{PYTHON} -m piptools sync requirements.txt')

@task(aliases=["i", "r", "reset", "reinstall"])
def install(c):
    # Reinstall all dependencies from requirements.txt, upgrading them if necessary.
    print("Reinstalling dependencies from requirements.txt...")
    with c.cd(PROJECT_ROOT):
        c.run(f'"{PIP}" install --upgrade -r requirements.txt')

@task(aliases=["extract", "e", "ef"])
def extract_features(c):
    # Extract audio features from the dataset and save them to a compressed .npz file.
    print("Extracting audio features to compressed .npz file...")
    with c.cd(TRAINING):
        c.run(f'"{PYTHON}" extract_features.py')

@task(aliases=["check_labels", "cl"])
def labels(c):
    # Check the labels in the feature dataset manifest for potential mislabelling.
    print("Checking labels in feature dataset manifest...")
    with c.cd(TRAINING):
        c.run(f'"{PYTHON}" check.py --labels')

@task(pre=[extract_features, train, convert, validate_tflite], aliases=["fp", "pl", "pipeline"])
def full_pipeline(_):
    # Run the full pipeline: extract features, train the model, and convert to TFLite.
    print("Running full pipeline: extract features, train model, convert to TFLite...")
    pass

@task(aliases=["plt", "graph_latest", "gl"])
def plot_latest(c):
    # Plot the learning curves from the most recent training report.
    print("Plotting learning curves from the most recent training report...")
    with c.cd(TRAINING):
        c.run(f'"{PYTHON}" chart.py --latest')

@task(aliases=["ph", "graph_history", "gh"])
def plot_history(c):
    # Plot the evolution of final test/validation metrics across all training reports.
    print("Plotting history of final metrics across all training reports...")
    with c.cd(TRAINING):
        c.run(f'"{PYTHON}" chart.py --history')