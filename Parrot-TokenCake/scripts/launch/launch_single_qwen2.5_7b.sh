#!/bin/sh

mkdir -p log

set -x SIMULATE_NETWORK_LATENCY_PRT 0

echo "Start ServeCore server ..."
python3 -m parrot.serve.http_server --config_path sample_configs/core/localhost_serve_core.json --log_dir log/ --log_filename core_1_qwen2.5_7b.log &

sleep 1

echo "Start one single Qwen2.5 7B server ..."
python3 -m parrot.engine.http_server --config_path sample_configs/engine/qwen2.5-7b-instruct-local.json --log_dir log/ --log_filename engine_1_qwen2.5_7b.log &

sleep 15

echo "Successfully launched Parrot runtime system."
