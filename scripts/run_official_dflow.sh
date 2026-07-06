#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-woosh}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WOOSH_ROOT="${WOOSH_ROOT:-${REPO_ROOT}/Woosh-main}"

cd "${WOOSH_ROOT}"
exec conda run -n "${CONDA_ENV}" python test_Woosh-DFlow.py
