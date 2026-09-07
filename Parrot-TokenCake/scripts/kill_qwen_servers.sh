#!/bin/sh
set -e

echo "Stop Qwen Parrot servers ..."

PIDS=$(ps -ef | grep -E "parrot\.(serve|engine)\.http_server|qwen2\.5" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "No Qwen-related Parrot servers found."
    exit 0
fi

echo "$PIDS" | xargs kill -9
echo "Successfully killed Qwen-related Parrot servers."
