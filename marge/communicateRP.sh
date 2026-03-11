#!/bin/sh
if command -v timeout >/dev/null 2>&1; then
    exec timeout 2s ssh -o BatchMode=yes -o ConnectTimeout=2 -o ConnectionAttempts=1 "root@$1" "$2"
elif command -v gtimeout >/dev/null 2>&1; then
    # On MacOS, the timeout command is called gtimeout and is provided by coreutils. If it's available, use it instead of timeout.
    # Use homebrew to install coreutils: brew install coreutils
    exec gtimeout 2s ssh -o BatchMode=yes -o ConnectTimeout=2 -o ConnectionAttempts=1 "root@$1" "$2"
else
    echo "Error: timeout command not found. Please install coreutils to get the timeout command." >&2
fi
