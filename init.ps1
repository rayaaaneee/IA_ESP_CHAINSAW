if (-Not (Test-Path -Path "venv" -PathType Container)) {
    Write-Host "Creating a new virtual environment with Python 3.11..."
    py -3.11 -m venv venv
} else {
    Write-Host "Virtual environment already exists. Skipping creation."
}

Write-Host "`nUpgrading pip and installing dependencies..."
& .\venv\Scripts\python.exe -m pip install --upgrade pip
& .\venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "`nSetup complete.`nYou can now run 'invoke train' to start training the model."