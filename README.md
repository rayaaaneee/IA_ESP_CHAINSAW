# IA_ESP_CHAINSAW

## PYTHON 3.11.X REQUIRED

This program have to be runned into ESP32

## PATHWAY

Install Python 3.11.X and add it to your PATH environment variable.
After, run the following script `init` to create a virtual environment and install the dependencies.

### Windows

#### PowerShell

```powershell
# Example for Windows PowerShell
$env:PATH = "C:\Python311;" + $env:PATH
./init.ps1
```

#### Command Prompt

```cmd
# Example for Windows Command Prompt
set PATH=C:\Python311;%PATH%
.\init.bat
```

### Linux

```bash
# Example for Linux
export PATH="/usr/bin/python3.11:$PATH"
./init.sh
```