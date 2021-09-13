#!/bin/bash

PR=$(echo ${GITHUB_REF} | cut -f3 -d/)

echo "Scanning file headers"
echo "GitHub API: ${GITHUB_API_URL}/repos/${GITHUB_REPOSITORY}/pulls/${PR}/comments"

function add_comment_for() {
  FILE=$(echo $1 | cut -f1 -d:)
  LINE=$(echo $1 | cut -f2 -d:)
  MSG=$(echo $1 | cut -f3 -d:)
  echo "Adding comment '${MSG}' to ${FILE}:${LINE} ..."
  curl \
    -X POST \
    -H "Authorization: token ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github.v3+json" \
    ${GITHUB_API_URL}/repos/${GITHUB_REPOSITORY}/pulls/${PR}/comments \
    -d "{
      \"body\":\"${MSG}\",
      \"path\":\"${FILE}\",
      \"side\":\"RIGHT\",
      \"line\":\"${LINE}\"
    }"
}

rm comments
for changed_file in ${GITHUB_CHANGED_FILES}; do
  ${GITHUB_WORKSPACE}/CMSIS/Utilities/check_header.sh ${changed_file} >> comments
done

while IFS="" read -r p || [ -n "$p" ]; do
  add_comment_for "$p"
done < comments

exit 0
