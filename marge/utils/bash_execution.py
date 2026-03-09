import os
import platform
import shutil
import subprocess
from pathlib import Path

import marge.configs.hw_config as hw


def _resolve_windows_bash():
    """
    Locate a usable bash executable on Windows.

    Prefer an executable already available on PATH so custom or system-wide Git
    installations work without extra configuration. If PATH does not contain a
    bash executable, probe the most common Git for Windows installation
    directories before failing.
    """
    for candidate in ("bash.exe", "bash"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    search_roots = []
    for env_var in ("LOCALAPPDATA", "ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(env_var)
        if root:
            search_roots.append(Path(root))

    for root in search_roots:
        for relative in ("Programs/Git/usr/bin/bash.exe", "Programs/Git/bin/bash.exe", "Git/usr/bin/bash.exe", "Git/bin/bash.exe"):
            candidate = root / relative
            if candidate.exists():
                return str(candidate)

    raise FileNotFoundError(
        "Unable to locate a Windows bash executable. Add Git Bash to PATH or configure hw.bash_override."
    )


def get_bash_launcher():
    """
    Resolve the launcher used for bash/script execution.

    A configured hardware override wins. Otherwise we fall back to a small
    platform-specific default so callers do not have to duplicate this logic.
    """
    if getattr(hw, "bash_override", ""):
        return str(hw.bash_override)

    system = platform.system()
    if system == "Linux":
        return "gnome-terminal"
    if system == "Windows":
        return _resolve_windows_bash()
    if system == "Darwin":
        return str(Path("/Applications/Terminal.app"))
    raise RuntimeError(f"Unsupported operating system: {system}")


def build_bash_command(script_path, *script_args):
    """
    Build the concrete subprocess command for the current platform.

    macOS `.app` bundles cannot be executed directly with `subprocess.run`, so
    the default Terminal.app launcher is translated to a real shell for these
    blocking script executions.
    """
    launcher = get_bash_launcher()
    if platform.system() == "Darwin" and launcher.endswith(".app"):
        return ["/bin/bash", str(script_path), *[str(arg) for arg in script_args]]
    return [launcher, "--", str(script_path), *[str(arg) for arg in script_args]]


def BashExecution(script_path, *script_args, timeout=None, **kwargs):
    """
    Execute a shell-accessible script through the configured terminal/bash launcher.

    This keeps launcher resolution and the wrapper command structure in one
    place so callers do not have to duplicate it.
    """
    command = build_bash_command(script_path, *script_args)
    return subprocess.run(command, timeout=timeout, **kwargs)
