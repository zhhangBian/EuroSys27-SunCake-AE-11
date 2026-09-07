# SunCake Environment Setup

Use Linux, Python 3.12, `uv`, and an NVIDIA driver supporting CUDA 13.0.
This installs the local SunCake Python code with precompiled vLLM extensions;
local CUDA and Rust compilation are not required.

Run from the artifact root:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install -r requirements/build/cuda.txt -r requirements/agent.txt

VLLM_USE_PRECOMPILED=1 \
  VLLM_PRECOMPILED_WHEEL_COMMIT=0b3ba88f165976e77ca5e6a7a3f5bba4562b80af \
  VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
  uv pip install --no-build-isolation -e .

export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_SAMPLER=0
```

The wheel is pinned to this source tree's upstream base. The `-e .` installation
uses Python files from the current directory: edit them and restart the server
to apply changes. C++/CUDA changes require rebuilding the corresponding extensions.

DeepGEMM files may be bundled in the wheel, but the first two environment variables
disable its use. The last selects the PyTorch sampler to avoid FlashInfer's CUDA
JIT compilation. Set all three in each shell used to launch experiments.
Model weights are supplied separately.

For CPU-only plotting, use the environment commands in
[README.md](README.md#environment). Experiment commands are in
[scripts/exp/README.md](scripts/exp/README.md).
