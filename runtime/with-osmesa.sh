#!/usr/bin/env bash
# Real MuJoCo physics and CPU OSMesa workers, as in the completed Walker runs.
# This wrapper selects neither a GPU nor a training budget; pass the actual command.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  printf '%s\n' 'Usage: bash runtime/with-osmesa.sh python -m spikedreamer.train ...' >&2
  exit 2
fi
if [ -n "${LD_PRELOAD:-}" ]; then
  printf '%s\n' 'Refusing inherited LD_PRELOAD in the parent. Use SPIKEDREAMER_ENV_LD_PRELOAD for render workers only.' >&2
  exit 2
fi
if [ -n "${SPIKEDREAMER_ENV_LD_PRELOAD:-}" ] && [ ! -r "$SPIKEDREAMER_ENV_LD_PRELOAD" ]; then
  printf '%s\n' 'SPIKEDREAMER_ENV_LD_PRELOAD is not readable.' >&2
  exit 2
fi

export SPIKEDREAMER_ENV_PROCESS=1
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export LP_NUM_THREADS=1
export PYTHONFAULTHANDLER=1
exec "$@"
