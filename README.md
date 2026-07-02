# IA_ESP_CHAINSAW

This program have to be runned into ESP32

[There's code tools (Arduino, `.ino`)](https://github.com/Nixmus/Esp32)

## PYTHON INSTALLATION PATHWAY

Install Python 3.11.X and add it to your PATH environment variable.
Afterwards, run the following script `init` to create a virtual environment and install the dependencies.

### Windows

Follow this link to install Python 3.11.X: [Python 3.11.X](https://www.python.org/downloads/release/python-3110/).

#### PowerShell

```powershell
# Example for Windows PowerShell
$env:PATH = "C:\Python311;" + $env:PATH
./init.ps1
```

#### Command Prompt

```bat
:: Example for Windows Command Prompt
set PATH=C:\Python311;%PATH%
.\init.bat
```

#### WSL / Git Bash

```bash
export PATH="/usr/bin/python3.11:$PATH"
./init.sh
```

### Linux

```bash
# Example for Linux
export PATH="/usr/bin/python3.11:$PATH"
./init.sh
```

## Dependencies Management

If you are adding a library to the `requirements.in` file (base requirements), you have to run the `invoke dependencies` command to update the `requirements.txt` file using **pip-tools**.

## VSCODE 

### Extensions

Install the recommended extensions for VSCODE, notably `PlatformIO IDE`, which is required to build and upload the firmware to the ESP32.
Besides, you can install the following extensions to improve your experience:
- `Python` by Microsoft
- `Pylance` by Microsoft
- `C/C++` by Microsoft
- `C/C++ Advanced Lint` by Jean Pierre Boudra
- `CMake Tools` by Microsoft
- `CMake` by twxs
- `CMake Language Support` by vector-of-bool
- `CMake Tools Helper` by vector-of-bool
- `CMake Tools Extension Pack` by vector-of-bool

### Workspace

Open the workspace file `IA_ESP_CHAINSAW.code-workspace` to have a better experience (File > Open Workspace from File).

## VENV

Afterwards, you can activate the virtual environment and run the main script.

### Windows

#### PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

#### Command Prompt

```bat
.\venv\Scripts\Activate.bat
```

#### WSL / Git Bash

```bash
source venv/Scripts/activate
```

### Linux

```bash
source venv/bin/activate
```
