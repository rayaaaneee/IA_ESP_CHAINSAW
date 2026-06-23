#!/bin/bash

if [ ! -d "venv" ]; then
    echo "Creating a new virtual environment with Python 3.11..."
    python3.11 -m venv venv
else
    echo "Virtual environment already exists. Skipping creation."
fi

echo "Upgrading pip and installing dependencies..."
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt

echo "Setup complete. You can now run 'invoke train' to start training the model."
