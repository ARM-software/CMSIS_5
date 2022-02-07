#!/bin/bash

FILE=$1
RESULT=0

FILE_DATE=$(grep -E '@date\s+([0-9]{2}\. \w+ [0-9]{4})' ${FILE} | sed -E 's/^.*@date\s+([0-9]{2}\. \w+ [0-9]{4}).*/\1/')
if [[ ! -z $FILE_DATE ]]; then
  AUTHOR_DATE=$(git log -1 --pretty="format:%ad" --date="format:%d. %B %Y" ${FILE})
  if [[ $AUTHOR_DATE != $FILE_DATE ]]; then
    FILE_DATE_LINE=$(grep -En "@date.*${FILE_DATE}" ${FILE} | cut -f1 -d:)
    echo "${FILE}:${FILE_DATE_LINE}:Please update file date to '$AUTHOR_DATE'." >&2
    RESULT=1
  fi
fi

FILE_VERSION=$(grep -E '@version\s+V?([0-9]+\.[0-9]+(\.[0-9]+)?)' ${FILE} | sed -E 's/^.*@version\s+V?([0-9]+\.[0-9]+(\.[0-9]+)?).*/\1/')
if [[ ! -z $FILE_VERSION ]]; then
  FILE_VERSION_LINE=$(grep -En "@version.*${FILE_VERSION}" ${FILE} | cut -f1 -d:)
  AUTHOR_REV=$(git log -1 --pretty="format:%H" -- ${FILE})
  PARENT_REV=$(git log -1 --pretty="format:%P" -- ${FILE})
  BLAME_REV=$(git blame ${PARENT_REV}..${AUTHOR_REV} -l -L ${FILE_VERSION_LINE},${FILE_VERSION_LINE} ${FILE} | sed -E 's/^([[:alnum:]]+).*/\1/')
  if [[ $AUTHOR_REV != $BLAME_REV ]]; then
    echo "${FILE}:${FILE_VERSION_LINE}:Please increment file version." >&2
    RESULT=1
  fi
fi

exit $RESULT
