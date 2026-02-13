"""
Utilities for running terminal/shell commands in a platform-aware way.
Handles macOS (empty bash_path), Linux (gnome-terminal), and Windows (Git bash).
"""

import subprocess

import marge.configs.hw_config as hw


def run_terminal_command(cmd_args):
    """
    Run a terminal command, handling macOS case where bash_path may be empty.

    Args:
        cmd_args: List of command arguments. If bash_path is empty (macOS),
                  runs the command directly. Otherwise uses bash_path as the
                  terminal emulator.

    Returns:
        subprocess.CompletedProcess or None
    """
    if not hw.bash_path or str(hw.bash_path).strip() == "":
        # macOS: bash_path is empty, run command directly
        # Extract the actual command (everything after "--")
        if "--" in cmd_args:
            cmd_start = cmd_args.index("--") + 1
            actual_cmd = cmd_args[cmd_start:]
        else:
            actual_cmd = cmd_args

        # For interactive commands (with "exec bash"), use osascript to open Terminal
        cmd_str = " ".join(actual_cmd)
        if "exec bash" in cmd_str or "sudo" in cmd_str:
            # Use osascript to open Terminal.app with the command
            applescript = f'''
            tell application "Terminal"
                activate
                do script "{cmd_str.replace('"', '\\"')}"
            end tell
            '''
            return subprocess.run(["osascript", "-e", applescript])
        else:
            # Run directly
            return subprocess.run(actual_cmd)
    else:
        # Linux/Windows: use bash_path as terminal emulator
        return subprocess.run(cmd_args)
