#!/usr/bin/env bash
# Use the installed environment without resolving or replacing its dependencies.
set -euo pipefail
SM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$SM_ROOT/.venv/bin/python" ]]; then
  echo "Missing .venv/bin/python; see README.md for environment setup." >&2
  exit 1
fi
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX
export PATH="$SM_ROOT/.venv/bin:$PATH"
export TMPDIR="$SM_ROOT/.cache/tmp"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "$TMPDIR"
cd "$SM_ROOT"
exec "$SM_ROOT/.venv/bin/python" "$@"
