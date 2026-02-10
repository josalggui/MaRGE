[![](https://img.shields.io/badge/marcos__client-blue)](https://github.com/vnegnev/marcos_client)
[![](https://img.shields.io/badge/marcos__server-blue)](https://github.com/vnegnev/marcos_server)
[![](https://img.shields.io/badge/marcos__extras-blue)](https://github.com/vnegnev/marcos_extras)

# MaRGE (MaRCoS Graphical Environment)

🚀 **Version 1.0.0 coming soon — stay tuned!**

**MaRGE** is a Python-based graphical environment for interacting with the **MaRCoS MRI research system**. It provides a user-friendly GUI for running MRI sequences, configuring parameters, and managing experiments without needing to work directly with low-level control code.

MaRGE is designed for MRI researchers, developers, and students working with the MaRCoS platform.

---

## Index
1. Preview
2. Features
3. General requirements
4. Installation with pip
5. Installation from source (For developers)
6. Full Installation (MaRGE + MaRCoS)
7. Documentation
8. Additional notes

## 1. Preview

![MaRGE GUI](marge/resources/images/main_clean.png)

---

## 2. Features

🧲 Graphical interface for MaRCoS MRI experiments

⚙️ Sequence configuration and execution

🧪 Research-oriented workflow

🧩 Extensible sequence architecture

🐍 Fully Python-based

---

## 3. General requirements

Before installing MaRGE, make sure your system meets the following minimum requirements. The software is primarily tested and supported on Ubuntu, but other platforms may work with some limitations.

1. **Ubuntu 22.04.5 LTS** with **Python 3.10**
2. **Windows 10** with **Python 3.13**
3. **Windows 11** with **Python 3.13**
3. **Internet connection**

Without a configured MaRCoS + Red Pitaya setup, MaRGE can still be launched, but only **Trial Mode** will be available (no hardware acquisition).

---

## 4. Installation with pip
Install MaRGE with pip if you only need to run the GUI and do not plan to modify the source code or add custom sequences.

    Note: Tyger capabilities are not supported from pip installation.

1. Go to your project folder.Create and activate a virtual environment:

* Ubuntu
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
* Windows
  ```bash
  python -m venv venv
  source venv/Scripts/activate
  ```

2. Install marge-mri
   ```bash
   pip install marge-mri==1.0.0b1
   ```

3. Launch the GUI from terminal:
   ```bash
     marge-mri
   ```

---

## 5. Installation from source (Developer mode)

Use this method if you want to modify the codebase or include your own sequences.

1. Go to the folder where you want to create the project and clone the repo
    ```bash
    git clone https://github.com/josalggui/MaRGE.git
    ```

2. Go into the created `MaRGE` folder. Create and activate a virtual environment.
Then, add the current folder (MaRGE) to Python's module search path:
* Ubuntu
    ```bash
    cd MaRGE
    python3 -m venv venv
    source venv/bin/activate
    export PYTHONPATH=$(pwd)
    ```
* Windows
    ```bash
    cd MaRGE
    python -m venv venv
    source venv/Scripts/activate
    export PYTHONPATH=$(pwd)
    ```
3. Install requirements
    ```bash
    pip install -r requirements.txt
    ```
4. Go into `MaRGE/marge` folder and run the `main.py`.
* Ubuntu
    ```bash
    cd marge
    python main.py
    ```
* Windows
    ```bash
    cd marge
    python3 main.py
    ```

### Notes
When you execute `main.py` additional folders are created to save results, configurations, or calibrations.
These folders are created in the directory where you run `main.py`.
It is highly recommended to run `main.py` from `MaRGE/marge` folder.

---

## 6. Full Installation (MaRGE + MaRCoS Setup)

For a complete installation including MaRCoS configuration, hardware setup, and developer options, please follow the detailed step-by-step guide in the Wiki:

👉 [Full Installation Guide](https://github.com/josalggui/MaRGE/wiki/Setting-up-MaRGE-and-MaRCoS-from-scratch)

---

## 7. Documentation

📖 Wiki: https://github.com/josalggui/MaRGE/wiki
 (under development)

📚 Documentation site: https://josalggui.github.io/MaRGE/
 (under development)

📦 PyPI package: https://pypi.org/project/marge-mri/

## 8. Additional notes

### 1. cupy-cuda12x module related error

During the installation of the requirements, the following error may appear:

`ERROR: No matching distribution found for cupy-cuda12x`

This usually means your Python environment is not compatible with the prebuilt CuPy CUDA 12 wheels. It is **not typically caused by a missing CUDA Toolkit installation**.

**Common causes and fixes:**

- **Unsupported Python version** — CuPy wheels are only published for specific Python versions. Check your version:

  ```bash
  python --version
  ```
If you are using Python 3.14 (or a newer unsupported version), install Python 3.13.0 and recreate your virtual environment.

### 2. CuPy / CUDA dependency and Tyger capability

CuPy is used in the postprocessing toolbox to accelerate the Algebraic Reconstruction Technique (ART) with GPU computation. The code attempts to import CuPy at runtime; if the import fails, ART automatically falls back to a CPU implementation.

In the current version, with the introduction of the **Tyger capability**, the ART-based postprocessing workflow is generally no longer needed. As a result:

- CuPy is effectively optional
- CUDA Toolkit is not required for normal Tyger-based workflows
- Failing to install `cupy-cuda12x` will **not** break the pipeline
- The code will continue to run using CPU paths (or Tyger paths) instead

You may safely skip CuPy/CUDA installation unless you explicitly plan to use the legacy ART postprocessing toolbox.
