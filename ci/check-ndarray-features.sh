#!/usr/bin/env bash
# Assert how ndarray resolves for a set of cargo feature flags.
# Usage: ci/check-ndarray-features.sh <absent|std|no-std> [cargo feature flags...]
set -euo pipefail

expect="$1"
shift

# `-i ndarray --depth 0` prints only the ndarray node, as "ndarray vX.Y.Z|feat1,feat2".
# Dev-dependency edges are excluded so test-only crates cannot affect the result.
if ! line="$(cargo tree -q -e normal,build -i ndarray --depth 0 --format '{p}|{f}' "$@" 2>&1)"; then
  case "$line" in
    *"did not match any packages"*) line="" ;;
    *) printf '%s\n' "$line" >&2; exit 1 ;;
  esac
fi

if [ -z "$line" ]; then
  actual="absent"
elif [[ ",${line#*|}," == *",std,"* ]]; then
  actual="std"
else
  actual="no-std"
fi

if [ "$actual" != "$expect" ]; then
  echo "ndarray with [$*]: expected $expect, got $actual (${line:-not in graph})" >&2
  exit 1
fi
echo "ndarray with [$*]: $actual"
