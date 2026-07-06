#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_ROOT="${REPO_ROOT}/src/evaluation/predefined_config"
DATA_ROOT="${REPO_ROOT}/data"

if [[ ! -d "${CONFIG_ROOT}" ]]; then
  echo "Config directory not found: ${CONFIG_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Data directory not found: ${DATA_ROOT}" >&2
  exit 1
fi

shopt -s nullglob

copied=0
for config_dir in "${CONFIG_ROOT}"/*; do
  [[ -d "${config_dir}" ]] || continue

  dataset_name="$(basename "${config_dir}")"
  config_file="${config_dir}/evaluation_config.json"
  target_dir="${DATA_ROOT}/${dataset_name}"

  if [[ ! -f "${config_file}" ]]; then
    echo "Missing evaluation_config.json in ${config_dir}" >&2
    exit 1
  fi

  if [[ ! -d "${target_dir}" ]]; then
    echo "Target data directory not found: ${target_dir}" >&2
    exit 1
  fi

  target_config_dir="${target_dir}/.evaluation_config"
  mkdir -p "${target_config_dir}"
  cp "${config_file}" "${target_config_dir}/evaluation_config.json"
  echo "Copied ${config_file} -> ${target_config_dir}/evaluation_config.json"
  copied=$((copied + 1))
done

echo "Delivered ${copied} evaluation config file(s)."
