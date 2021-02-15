#!/bin/bash
#
# This bash script run cbuild.sh for a given Flash Algorithm project
#
# Pre-requisites:
# - bash shell (for Windows: install git for Windows)
# - source <CMSIS Build installation>/etc/setup

version=0.1.0

# header
echo "($(basename "$0")): Build Flash Algorithm project $version (C) 2020 ARM"

usage() {
  echo "Usage:"
  echo "  FlashBuild.sh <projectname>.cprj [<toolchain>]"
  echo ""
  echo "  <projectname>.cprj : Flash Algorithm project filename"
  echo "  <toolchain>        : AC6 | GCC   default is AC6"
}

toolchain="AC6"

# arguments
for i in "$@"
do
  case $i in
    *.cprj)
      filename=$(basename "$i" .cprj)
      shift
    ;;
    AC6)
      toolchain="AC6"
      shift
    ;;
    GCC)
      toolchain="GCC"
      shift
    ;;
    --h)
      usage
      exit 0
    ;;
    ?|-*|--*)
      usage
      exit 0
    ;;
  esac
done

if [ -z ${filename+x} ]
  then
  echo "error: missing required argument <projectname>.cprj"
  usage
  exit 1
fi

if [ ! -f "$filename.cprj" ]
  then
  echo "error: Flash Algorithm project file $filename.cprj not found"
  exit 1
fi

# call cbuild project
cbuild.sh "$filename".cprj --toolchain="$toolchain"
if [ $? -ne 0 ]
  then
  echo "Falsh Algorithm project $filename.cprj failed"
  exit 1
fi

if [ $toolchain == "AC6" ]
  then
  cp ./Out/"$filename".axf ./"$filename".FLM
fi

if [ $toolchain == "GCC" ]
  then
  cp ./Out/"$filename".elf ./"$filename".FLM
fi

exit 0
