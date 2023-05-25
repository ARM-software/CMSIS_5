#!/usr/bin/env bash

set -o pipefail

DIRNAME=$(dirname $(realpath $0))
REQUIRED_GEN_PACK_LIB="0.8.3"

############ gen-pack library ###########

function install_lib() {
  local URL="https://github.com/Open-CMSIS-Pack/gen-pack/archive/refs/tags/v$1.tar.gz"
  local STATUS=$(curl -sLI "${URL}" | grep "^HTTP" | tail -n 1 | cut -d' ' -f2 || echo "$((600+$?))")
  if [[ $STATUS -ge 400 ]]; then
    echo "Wrong/unavailable gen-pack lib version '$1'!" >&2
    echo "Check REQUIRED_GEN_PACK_LIB variable."  >&2
    echo "For available versions see https://github.com/Open-CMSIS-Pack/gen-pack/tags." >&2
    exit 1
  fi
  echo "Downloading gen-pack lib version '$1' to '$2' ..."
  mkdir -p "$2"
  curl -L "${URL}" -s | tar -xzf - --strip-components 1 -C "$2" || exit 1
}

function load_lib() {
  if [[ -d ${GEN_PACK_LIB} ]]; then
    . "${GEN_PACK_LIB}/gen-pack"
    return 0
  fi
  local GLOBAL_LIB="/usr/local/share/gen-pack/${REQUIRED_GEN_PACK_LIB}"
  local USER_LIB="${HOME}/.local/share/gen-pack/${REQUIRED_GEN_PACK_LIB}"
  if [[ ! -d "${GLOBAL_LIB}" && ! -d "${USER_LIB}" ]]; then
    echo "Required gen-pack lib not found!" >&2
    install_lib "${REQUIRED_GEN_PACK_LIB}" "${USER_LIB}"
  fi

  if [[ -d "${GLOBAL_LIB}" ]]; then
    . "${GLOBAL_LIB}/gen-pack"
  elif [[ -d "${USER_LIB}" ]]; then
    . "${USER_LIB}/gen-pack"
  else
    echo "Required gen-pack lib is not installed!" >&2
    exit 1
  fi
}

load_lib
find_linkchecker

#########################################

check_links "${DIRNAME}/../Documentation/index.html" "${DIRNAME}"
