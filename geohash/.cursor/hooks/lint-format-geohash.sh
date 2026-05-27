#!/usr/bin/env bash
set -euo pipefail

input=$(cat)
file_path=$(echo "$input" | jq -r '.file_path // empty')
workspace_root=$(echo "$input" | jq -r '.workspace_roots[0] // empty')

if [[ -z "$file_path" ]]; then
  exit 0
fi

if [[ -n "$workspace_root" ]]; then
  project_root="${workspace_root%/}"
else
  project_root="$(cd "$(dirname "$0")/../.." && pwd)"
fi

geohash_pkg="${project_root}/geohash"
case "$file_path" in
  "$geohash_pkg"/*) ;;
  *) exit 0 ;;
esac

cd "$project_root"
just format
just lint

exit 0
