#!/usr/bin/env bash
set -euo pipefail

AE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AE_PYTHON="${SUNCAKE_PYTHON:-${AE_ROOT}/.venv/bin/python}"
if [[ ! -x "${AE_PYTHON}" ]]; then
    printf 'Build the artifact in .venv or set SUNCAKE_PYTHON to its Python executable.\n' >&2
    exit 1
fi
cd -- "${AE_ROOT}"
exec "${AE_PYTHON}" -m ae.server "$@"
