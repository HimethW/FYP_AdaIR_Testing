#!/bin/bash
# Compact fingerprint of the datasets, to check that two servers hold identical data. Elgin's copy is downloaded
# there from the original sources (tools/prepare_data.py + tools/build_mixtures.py) instead of being copied from
# qbits, so this is how we know the baseline trains and validates on exactly the same files as ours.
# One line per dataset folder (data/<a>/<b>): file count, total bytes, and a sha256 over the sorted
# "md5  path" list of every file in it.
#   bash tools/data_digest.sh data > digest.txt          # then diff the two servers' outputs
#   bash tools/data_digest.sh data data/test/mix         # per-file md5 list of one folder, to find a mismatch
set -euo pipefail
export LC_ALL=C
root=${1:-data}
if [ $# -ge 2 ]; then
  (cd "$2" && find . -type f -print0 | sort -z | xargs -0 -r md5sum)
  exit
fi
for d in "$root"/*/*/; do
  d=${d%/}
  n=$(find "$d" -type f | wc -l)
  bytes=$(find "$d" -type f -printf '%s\n' | awk '{s += $1} END {print s + 0}')
  digest=$(cd "$d" && find . -type f -print0 | sort -z | xargs -0 -r md5sum | sha256sum | cut -c1-16)
  printf '%s\t%s\t%s\t%s\n' "${d#"$root"/}" "$n" "$bytes" "$digest"
done
