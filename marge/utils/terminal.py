"""
Utilities for running terminal/shell commands in a platform-aware way.
If hw.bash_path is set by the user, that value is used. Otherwise behavior is
auto-detected: Linux -> gnome-terminal, Windows -> Git bash, macOS -> run
directly or open Terminal.app for interactive commands.
"""

import getpass
import platform
import subprocess
from pathlib import Path

import marge.configs.hw_config as hw


def _default_bash_path():
    """Platform-specific default when user has not set hw.bash_path."""
    system = platform.system()
    if system == "Linux":
        return "gnome-terminal"
    if system == "Windows":
        username = getpass.getuser()
        return Path(rf"C:\Users\{username}\AppData\Local\Programs\Git\usr\bin\bash.exe")
    if system == "Darwin":
        return ""
    raise RuntimeError(f"Unsupported operating system: {system}")


def run_terminal_command(cmd_args):
    """
    Run a terminal command. Uses hw.bash_path if set (non-empty); otherwise
    uses platform auto-detection (gnome-terminal on Linux, Git bash on Windows,
    direct/osascript on macOS).

    Args:
        cmd_args: List of command arguments, typically [bash_path, "--", ...].
                  The first element is ignored when we substitute the effective
                  bash path.

    Returns:
        subprocess.CompletedProcess or None
    """
    user_set = hw.bash_path and str(hw.bash_path).strip() != ""
    effective = hw.bash_path if user_set else _default_bash_path()

    # Extract the actual command (everything after "--")
    if "--" in cmd_args:
        cmd_start = cmd_args.index("--") + 1
        actual_cmd = cmd_args[cmd_start:]
    else:
        actual_cmd = cmd_args

    if not effective or str(effective).strip() == "":
        # macOS-style: no terminal emulator, run command directly or open Terminal.app
        cmd_str = " ".join(actual_cmd)
        if "exec bash" in cmd_str or "sudo" in cmd_str:
            applescript = f'''
            tell application "Terminal"
                activate
                do script "{cmd_str.replace('"', '\\"')}"
            end tell
            '''
            return subprocess.run(["osascript", "-e", applescript])
        return subprocess.run(actual_cmd)

    # User set bash_path or platform default is a terminal emulator
    return subprocess.run([effective, "--"] + actual_cmd)
