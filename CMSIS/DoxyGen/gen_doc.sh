#!/usr/bin/env bash
# Version: 2.3
# Date: 2023-06-06
# This bash script generates CMSIS-Core documentation
#
# Pre-requisites:
# - bash shell (for Windows: install git for Windows)
# - doxygen 1.9.6
# - mscgen 0.20

set -o pipefail

# Set version of gen pack library
# For available versions see https://github.com/Open-CMSIS-Pack/gen-pack/tags.
# Use the tag name without the prefix "v", e.g., 0.7.0
REQUIRED_GEN_PACK_LIB="0.8.4"

DIRNAME=$(dirname $(readlink -f $0))
GENDIR=../Documentation
REQ_DXY_VERSION="1.9.6"
REQ_MSCGEN_VERSION="0.20"

############ DO NOT EDIT BELOW ###########

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
find_git
find_doxygen "${REQ_DXY_VERSION}"
find_utility "mscgen" "-l | grep 'Mscgen version' | sed -r -e 's/Mscgen version ([^,]+),.*/\1/'" "${REQ_MSCGEN_VERSION}"

if [ -z "${VERSION_FULL}" ]; then
  VERSION_FULL=$(git_describe "v")
fi

pushd "${DIRNAME}" > /dev/null

echo "Generating documentation ..."

function generate() {
  pushd $1 > /dev/null

  projectName=$(grep -E "PROJECT_NAME\s+=" $1.dxy.in | sed -r -e 's/[^"]*"([^"]+)".*/\1/')
  projectNumberFull="$2"
  if [ -z "${projectNumberFull}" ]; then
    projectNumberFull=$(grep -E "PROJECT_NUMBER\s+=" $1.dxy.in | sed -r -e 's/[^"]*"[^0-9]*(([0-9]+\.[0-9]+(\.[0-9]+)?(-.+)?)?)".*/\1/')
  fi
  if [ -z "${projectNumberFull}" ]; then
    projectNumberFull="$(git rev-parse --short HEAD)"
  fi
  projectNumber="${projectNumberFull%+*}"
  datetime=$(date -u +'%a %b %e %Y %H:%M:%S')
  year=$(date -u +'%Y')

  sed -e "s/{projectNumber}/${projectNumber}/" $1.dxy.in > $1.dxy

  # git_changelog -f html -p "v" > src/history.txt

  echo "\"${UTILITY_DOXYGEN}\" $1.dxy"
  "${UTILITY_DOXYGEN}" $1.dxy

  mkdir -p "${DIRNAME}/${GENDIR}/$1/html/search/"
  cp -f "${DIRNAME}/Doxygen_Templates/search.css" "${DIRNAME}/${GENDIR}/$1/html/search/"
  cp -f "${DIRNAME}/Doxygen_Templates/navtree.js" "${DIRNAME}/${GENDIR}/$1/html/"

  sed -e "s/{datetime}/${datetime}/" "${DIRNAME}/Doxygen_Templates/footer.js.in" \
    | sed -e "s/{year}/${year}/" \
    | sed -e "s/{projectName}/${projectName}/" \
    | sed -e "s/{projectNumber}/${projectNumber}/" \
    | sed -e "s/{projectNumberFull}/${projectNumberFull}/" \
    > "${DIRNAME}/${GENDIR}/$1/html/footer.js"

  popd > /dev/null
}

generate "General" "${VERSION_FULL}"
generate "Core"
generate "Core_A"
generate "Driver"
generate "RTOS2"
generate "DSP"
generate "NN"
generate "View"
generate "Compiler"
generate "Toolbox"
generate "Stream"
generate "DAP"
generate "Zone"

popd > /dev/null

exit 0
