# `mooncake_agent` 环境使用说明

本文档说明如何使用本地 `mooncake_agent` conda 环境。该环境已经为 Mooncake 和真实的 `vllm_agent` 代码仓库配置完成。

注意：仓库里不再保留 `scripts/mooncake_agent/` 脚本文件；脚本内容已经直接放在本文档中。需要运行时，把对应代码块保存成你自己的 `.py` 或 `.sh` 文件后执行即可。

## 环境内容

| 组件 | 值 |
| --- | --- |
| Conda 环境 | `/home/youwei/anaconda3/envs/mooncake_agent` |
| vLLM 源码路径 | `/home/youwei/bzh/project/TokenCake/vllm_agent` |
| vLLM 安装方式 | `mooncake_agent` 中的 editable 映射 |
| Mooncake Python 包 | `mooncake-transfer-engine==0.3.0b4` |
| Mooncake 模块路径 | `/home/youwei/anaconda3/envs/mooncake_agent/lib/python3.12/site-packages/mooncake` |
| Torch/CUDA 栈 | 从 `vllm_me` 克隆而来；已验证 `torch==2.6.0+cu124` |
| 已验证模型 | `/home/youwei/bzh/model/Qwen/Qwen2.5-7B-Instruct` |

最重要的一点是：在这个环境中执行 `import vllm` 时，导入的是实际的 vLLM 代码仓库：

```bash
conda run -n mooncake_agent python -c "import vllm; print(vllm.__file__)"
```

预期输出应包含：

```text
/home/youwei/bzh/project/TokenCake/vllm_agent/vllm/__init__.py
```

## 激活环境

```bash
source /home/youwei/anaconda3/etc/profile.d/conda.sh
conda activate mooncake_agent
```

也可以不激活环境，直接用 `conda run` 执行单条命令。

## 快速验证命令

```bash
conda run -n mooncake_agent python -c "import vllm, mooncake.store, torch; import vllm._C; print(vllm.__file__); print(mooncake.store.__file__); print(torch.__version__, torch.cuda.is_available())"
```

## 文档内脚本使用方式

本文后面提供完整脚本源码。使用方式：

1. 从对应章节复制代码块。
2. 保存成建议的文件名，例如 `check_imports.py` 或 `run_qwen25_7b_vllm_smoke.sh`。
3. 对 `.sh` 文件执行 `chmod +x 文件名`。
4. 运行该 `.py` 或 `.sh` 文件。

下面的运行示例假设你已经把代码块保存成对应文件名。

## 导入检查脚本

建议保存为 `check_imports.py`。

运行方式：

```bash
conda run -n mooncake_agent python check_imports.py
```

成功时会输出：

```text
MOONCAKE_AGENT_IMPORTS_OK
```

脚本内容：

```python
import os
from pathlib import Path

import mooncake.store
import torch
import vllm
import vllm._C


def main() -> None:
    expected_root = Path(
        os.environ.get(
            "EXPECTED_VLLM_ROOT",
            "/home/youwei/bzh/project/TokenCake/vllm_agent",
        )
    ).resolve()
    vllm_file = Path(vllm.__file__).resolve()

    print("vllm:", vllm.__version__, vllm_file)
    print("vllm._C: ok")
    print("mooncake.store:", mooncake.store.__file__)
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())

    if not str(vllm_file).startswith(str(expected_root)):
        raise SystemExit(
            f"vLLM 指向错误: {vllm_file}; 预期在 {expected_root} 下"
        )

    print("MOONCAKE_AGENT_IMPORTS_OK")


if __name__ == "__main__":
    main()
```

## Qwen2.5-7B vLLM 生成测试脚本

建议保存为 `qwen25_7b_vllm_smoke.py`。

运行方式：

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  conda run -n mooncake_agent python qwen25_7b_vllm_smoke.py
```

如果 GPU 1 不空闲，请修改 `CUDA_VISIBLE_DEVICES=1`。

成功时会输出：

```text
QWEN_VLLM_GENERATION_OK
```

脚本内容：

```python
import argparse

import mooncake.store
import torch
from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/home/youwei/bzh/model/Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument(
        "--prompt",
        default=(
            "Give a one-sentence proof that this vLLM and Mooncake "
            "environment works."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "torch",
        torch.__version__,
        "cuda_available",
        torch.cuda.is_available(),
        "device_count",
        torch.cuda.device_count(),
        flush=True,
    )
    print("mooncake_store", mooncake.store.__file__, flush=True)

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        enforce_eager=True,
    )

    outputs = llm.generate(
        [args.prompt],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )
    for output in outputs:
        print("PROMPT", output.prompt, flush=True)
        print("OUTPUT", output.outputs[0].text, flush=True)
    print("QWEN_VLLM_GENERATION_OK", flush=True)


if __name__ == "__main__":
    main()
```

## Qwen2.5-7B shell 包装脚本

建议保存为 `run_qwen25_7b_vllm_smoke.sh`，并与 `qwen25_7b_vllm_smoke.py` 放在同一目录。

运行方式：

```bash
chmod +x run_qwen25_7b_vllm_smoke.sh
./run_qwen25_7b_vllm_smoke.sh
```

脚本内容：

```bash
#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  source /home/youwei/anaconda3/etc/profile.d/conda.sh
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_NAME=${ENV_NAME:-mooncake_agent}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}

conda run -n "$ENV_NAME" python "$SCRIPT_DIR/qwen25_7b_vllm_smoke.py" "$@"
```

## MooncakeStore 接口冒烟测试脚本

这个测试会检查 vLLM 实际使用的 Mooncake 接口：导入 `mooncake.store.MooncakeDistributedStore`，并调用 `setup()`、`put()`、`get()`。

建议保存为 `mooncake_vllm_store_smoke.py`。

运行方式：

```bash
conda run -n mooncake_agent python mooncake_vllm_store_smoke.py
```

成功时会输出：

```text
VLLM_MOONCAKE_STORE_INTERFACE_OK
```

脚本内容：

```python
import argparse
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50123)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--global-segment-size", type=int, default=67_108_864)
    parser.add_argument("--local-buffer-size", type=int, default=16_777_216)
    parser.add_argument("--protocol", default="tcp")
    parser.add_argument("--device-name", default="")
    parser.add_argument("--metadata-server", default="P2PHANDSHAKE")
    parser.add_argument("--config-path")
    parser.add_argument("--master-log")
    parser.add_argument("--master-bin")
    return parser.parse_args()


def wait_for_port(host: str, port: int, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        with socket.socket() as sock:
            sock.settimeout(0.3)
            try:
                sock.connect((host, port))
                return
            except OSError as exc:
                last_error = exc
        time.sleep(0.2)
    raise RuntimeError(f"等待 mooncake_master 监听 {host}:{port} 超时: {last_error}")


def write_config(path: Path, args: argparse.Namespace) -> None:
    config = {
        "local_hostname": args.host,
        "metadata_server": args.metadata_server,
        "global_segment_size": args.global_segment_size,
        "local_buffer_size": args.local_buffer_size,
        "protocol": args.protocol,
        "device_name": args.device_name,
        "master_server_address": f"{args.host}:{args.port}",
    }
    path.write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    master_bin = args.master_bin or shutil.which("mooncake_master")
    if master_bin is None:
        raise SystemExit("找不到 mooncake_master，请在 mooncake_agent 环境中运行")

    with tempfile.TemporaryDirectory(prefix="mooncake_store_smoke_") as temp_dir:
        temp_path = Path(temp_dir)
        config_path = Path(args.config_path) if args.config_path else temp_path / "mooncake_smoke.json"
        master_log = Path(args.master_log) if args.master_log else temp_path / "mooncake_master_smoke.log"
        write_config(config_path, args)

        log_file = master_log.open("w")
        proc = subprocess.Popen(
            [master_bin, f"--port={args.port}", "--logtostderr=1"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_for_port(args.host, args.port)
            os.environ["MOONCAKE_CONFIG_PATH"] = str(config_path)

            from vllm.distributed.kv_transfer.kv_lookup_buffer.mooncake_store import MooncakeStore

            print("MOONCAKE_CONFIG_PATH", os.environ["MOONCAKE_CONFIG_PATH"])
            print("mooncake_master_log", master_log)
            store = MooncakeStore(None)
            key = "vllm_mooncake_smoke_tensor"
            value = torch.arange(16, dtype=torch.float32).reshape(4, 4)
            store.put(key, value)
            loaded = store.get(key)
            print("loaded", loaded)
            assert loaded is not None
            assert torch.equal(loaded.cpu(), value)
            print("VLLM_MOONCAKE_STORE_INTERFACE_OK")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            log_file.close()

        print("mooncake_master_log_tail")
        print("".join(master_log.read_text(errors="replace").splitlines(True)[-40:]))


if __name__ == "__main__":
    main()
```

## MooncakeStore shell 包装脚本

建议保存为 `run_mooncake_vllm_store_smoke.sh`，并与 `mooncake_vllm_store_smoke.py` 放在同一目录。

运行方式：

```bash
chmod +x run_mooncake_vllm_store_smoke.sh
./run_mooncake_vllm_store_smoke.sh --port 50123
```

脚本内容：

```bash
#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  source /home/youwei/anaconda3/etc/profile.d/conda.sh
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_NAME=${ENV_NAME:-mooncake_agent}

conda run -n "$ENV_NAME" python "$SCRIPT_DIR/mooncake_vllm_store_smoke.py" "$@"
```

## vLLM editable 映射修复脚本

该环境应该指向 `/home/youwei/bzh/project/TokenCake/vllm_agent`。如果输出意外指向了 `Mooncake-TokenCake/vllm_agent` 下的示例 checkout，请使用这个脚本修复。

建议保存为 `fix_vllm_editable_mapping.py`。

运行方式：

```bash
conda run -n mooncake_agent python fix_vllm_editable_mapping.py
```

如果真实 vLLM checkout 路径变了，可以传入新的目标路径：

```bash
conda run -n mooncake_agent python fix_vllm_editable_mapping.py \
  --target /home/youwei/bzh/project/TokenCake/vllm_agent
```

脚本内容：

```python
import argparse
import json
import re
import site
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="/home/youwei/bzh/project/TokenCake/vllm_agent",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = Path(args.target).resolve()
    vllm_pkg = target / "vllm"
    if not vllm_pkg.is_dir():
        raise SystemExit(f"找不到 vLLM package: {vllm_pkg}")

    site_paths = [Path(path) for path in site.getsitepackages()]
    patched = []
    for site_path in site_paths:
        for finder in site_path.glob("__editable___vllm_*_finder.py"):
            text = finder.read_text()
            text = re.sub(r"'vllm': '[^']*'", "'vllm': '" + str(vllm_pkg) + "'", text)
            finder.write_text(text)
            patched.append(finder)

        for direct_url in site_path.glob("vllm-*.dist-info/direct_url.json"):
            data = json.loads(direct_url.read_text())
            data["url"] = "file://" + str(target)
            data.setdefault("dir_info", {})["editable"] = True
            direct_url.write_text(json.dumps(data))
            patched.append(direct_url)

    for path in patched:
        print("patched", path)

    if not patched:
        raise SystemExit("没有找到 vLLM editable finder 或 direct_url.json")


if __name__ == "__main__":
    main()
```

## 普通 vLLM API Server shell 脚本

如果只运行普通 vLLM 服务，不启用 Mooncake KV transfer，可以使用这个脚本。

建议保存为 `run_vllm_api_server.sh`。

运行方式：

```bash
chmod +x run_vllm_api_server.sh
./run_vllm_api_server.sh
```

常用环境变量：

| 环境变量 | 默认值 |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | `1` |
| `MODEL` | `/home/youwei/bzh/model/Qwen/Qwen2.5-7B-Instruct` |
| `PORT` | `8000` |
| `MAX_MODEL_LEN` | `1024` |
| `GPU_MEMORY_UTILIZATION` | `0.45` |

例如改成 GPU 0 和端口 8001：

```bash
CUDA_VISIBLE_DEVICES=0 PORT=8001 ./run_vllm_api_server.sh
```

脚本内容：

```bash
#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  source /home/youwei/anaconda3/etc/profile.d/conda.sh
fi

ENV_NAME=${ENV_NAME:-mooncake_agent}
MODEL=${MODEL:-/home/youwei/bzh/model/Qwen/Qwen2.5-7B-Instruct}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1024}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

exec conda run -n "$ENV_NAME" \
  python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --trust-remote-code \
  "$@"
```

## 带 Mooncake KV Transfer 的 vLLM Server shell 脚本

先准备 Mooncake 配置文件，并设置 `MOONCAKE_CONFIG_PATH`：

```bash
export MOONCAKE_CONFIG_PATH=/path/to/mooncake.json
```

建议保存为 `run_vllm_mooncake_server.sh`。

启动 prefill / producer 侧：

```bash
chmod +x run_vllm_mooncake_server.sh
KV_ROLE=kv_producer PORT=8100 ./run_vllm_mooncake_server.sh
```

启动 decode / consumer 侧：

```bash
KV_ROLE=kv_consumer PORT=8200 ./run_vllm_mooncake_server.sh
```

脚本内容：

```bash
#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  source /home/youwei/anaconda3/etc/profile.d/conda.sh
fi

: "${MOONCAKE_CONFIG_PATH:?请先设置 MOONCAKE_CONFIG_PATH=/path/to/mooncake.json}"

ENV_NAME=${ENV_NAME:-mooncake_agent}
MODEL=${MODEL:-/home/youwei/bzh/model/Qwen/Qwen2.5-7B-Instruct}
PORT=${PORT:-8100}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1024}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
KV_CONNECTOR=${KV_CONNECTOR:-MooncakeStoreConnector}
KV_ROLE=${KV_ROLE:-kv_producer}
KV_TRANSFER_CONFIG='{"kv_connector":"'"$KV_CONNECTOR"'","kv_role":"'"$KV_ROLE"'"}'
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

exec conda run -n "$ENV_NAME" \
  python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --trust-remote-code \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  "$@"
```

## 已知注意事项

- `mooncake_agent` 是从 `vllm_me` 克隆出来的，然后在其中安装了 Mooncake。
- 进行 vLLM 生成测试时，请使用上面文档中的 Python 脚本内容保存成 `.py` 文件后运行，不要在命令行里临时 heredoc 生成脚本。
- Mooncake 构建需要环境内的 C++ 依赖（`gflags`、`glog`、`jsoncpp` 和 `yalantinglibs`）。这些依赖已经安装在 `mooncake_agent` 中。
- 如果从其他 Mooncake commit 重新构建，请在配合 vLLM 使用前重新运行导入检查和 MooncakeStore 冒烟测试。
