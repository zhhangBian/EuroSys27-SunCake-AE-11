#!/bin/sh
set -e

SERVER_URL="${1:-http://127.0.0.1:9000}"
MODEL_PATH="/home/youwei/bzh/model/Qwen/Qwen2.5-7B-Instruct"

echo "Testing Parrot common inference on ${SERVER_URL} with ${MODEL_PATH} ..."

curl -s "${SERVER_URL}/v1/common_inference" \
  -H 'Content-Type: application/json' \
  -X POST \
  -d "{
    \"model\": \"${MODEL_PATH}\",
    \"prompt\": \"Hello, my name is van, i am an artist, a performance artist.\",
    \"max_tokens\": 1,
    \"temperature\": 0
  }"

echo
