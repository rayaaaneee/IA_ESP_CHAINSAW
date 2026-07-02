@echo off

IF NOT EXIST "venv\" (
    echo Creating a new virtual environment with Python 3.11...
    echo.
    py -3.11 -m venv venv
) ELSE (
    echo Virtual environment already exists. Skipping creation.
)
echo.
echo Upgrading pip and installing dependencies...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo Setup complete. You can now run 'invoke train' to start training the model.
echo.
echo Finalized initialization! Please run 'inv list' to see all available commands.