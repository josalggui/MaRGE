[![](https://img.shields.io/badge/marcos__client-blue)](https://github.com/vnegnev/marcos_client)
[![](https://img.shields.io/badge/marcos__server-blue)](https://github.com/vnegnev/marcos_server)
[![](https://img.shields.io/badge/marcos__extras-blue)](https://github.com/vnegnev/marcos_extras)

# MaRGE (MaRCoS Graphical Environment)

## Installation with pip
A fast installation can be done using pip.
But note that pip installation only works with red pitayas configured with static IP 192.168.1.101.
You can install **MaRGE** using pip:

```bash
pip install marge-mri
```

Then, you can launch the GUI with:

```python
# Import marge
from marge import main

# Run MaRGE
main.MaRGE()
```

## Installation from source

Alternatively, if you prefer to download from source and have full access to the code, simply clone this repository:

```bash
git clone https://github.com/josalggui/MaRGE
```
Create your virtual environment and install all the requirements with

```bash
pip install -r requirements.txt
```

## Notes

This repository contains the Python code for the MaRCoS Graphical Environment (MaRGE), a system for magnetic resonance imaging research. The GUI provides a user-friendly interface to interact with the MaRCoS system.

Take a look at the MaRGE [Wiki](https://github.com/josalggui/MaRGE/wiki)! (under development)

Take a look at the MaRGE [Documentation](https://josalggui.github.io/MaRGE/)! (under development)

Take a look at the MaRGE [PyPi](https://pypi.org/project/marge-mri/)

### [Setting up a Red Pitaya](https://github.com/josalggui/MaRGE/wiki/Setting-up-Red-Pitaya)

### [Setting up MaRGE](https://github.com/josalggui/MaRGE/wiki/Setting-up-MaRGE)

### [Description of the GUI](https://github.com/josalggui/MaRGE/wiki/Interface-description)

### [Toolbars](https://github.com/josalggui/MaRGE/wiki/Toolbars)

### [Setting up the autocalibration](https://github.com/josalggui/MaRGE/wiki/Setting-up-autocalibration)

### [Setting up the localizer](https://github.com/josalggui/MaRGE/wiki/Setting-up-localizer)

### [Run custom sequences](https://github.com/josalggui/MaRGE/wiki/Run-custom-sequences)

### [Protocols](https://github.com/josalggui/MaRGE/wiki/Protocols)

### [Adding a New Sequence to the GUI](https://github.com/josalggui/MaRGE/wiki/Create-your-own-sequence)