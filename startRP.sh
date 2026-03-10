#!/bin/sh

../MaRGE/copy_bitstream.sh "$1" "$2"

if command -v timeout >/dev/null 2>&1; then
    exec timeout 2s ssh -o BatchMode=yes -o ConnectTimeout=2 -o ConnectionAttempts=1 "root@$1" "~/marcos_server &"
elif command -v gtimeout >/dev/null 2>&1; then
    exec gtimeout 2s ssh -o BatchMode=yes -o ConnectTimeout=2 -o ConnectionAttempts=1 "root@$1" "~/marcos_server &"
else
    exec ssh -o BatchMode=yes -o ConnectTimeout=2 -o ConnectionAttempts=1 "root@$1" "~/marcos_server &"
fi
