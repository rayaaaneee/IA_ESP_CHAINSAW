#!/bin/bash
set -e

# Define the correct Python and pip executables based on the operating system
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    IS_WIN=true
    PYTHON_EXEC="venv/Scripts/python"
    PIP_EXEC="venv/Scripts/pip"
else
    IS_WIN=false
    PYTHON_EXEC="./venv/bin/python"
    PIP_EXEC="./venv/bin/pip"
fi

if [ ! -d "venv" ]; then
    echo "Creating a new virtual environment with Python 3.11..."
    # Note: Ensure that Python 3.11 is installed and available in your PATH
    if [[ "$IS_WIN" == true ]]; then
        py -3.11 -m venv venv
    else
        python3.11 -m venv venv
    fi
else
    echo "Virtual environment already exists. Skipping creation."
fi

echo "Upgrading pip and installing dependencies..."
"$PYTHON_EXEC" -m pip install --upgrade pip
"$PYTHON_EXEC" -m pip install -r requirements.txt

echo -e "\nSetup complete."
echo -e "You can now run 'invoke train' to start training the model."
echo -e "Finalized initialization! Please run 'invoke list' to see available tasks.\n"
