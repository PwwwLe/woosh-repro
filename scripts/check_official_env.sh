#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-woosh}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WOOSH_ROOT="${WOOSH_ROOT:-${REPO_ROOT}/Woosh-main}"

echo "repo_root=${REPO_ROOT}"
echo "woosh_root=${WOOSH_ROOT}"
echo "conda_env=${CONDA_ENV}"

cd "${WOOSH_ROOT}"

conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/scripts/check_official_env.py"
