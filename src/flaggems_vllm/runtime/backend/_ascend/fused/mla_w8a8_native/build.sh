#!/usr/bin/env bash
# Rebuild the A2 helpers with the CANN SDK that supplies ccec and llvm-link.
set -euo pipefail
sdk=${CANN_SDK:-${ASCEND_HOME_PATH:?set CANN_SDK to the CANN toolkit root}}
cd "$(dirname "$0")"
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
asc="$sdk/aarch64-linux/asc"
for base in window m32 m16; do
    for arch in cube vec; do
        stem="${base}_${arch}"
        for mode in normal mixed; do
            flags=()
            if [[ "$mode" == mixed ]]; then
                flags=(-cce-enable-mix -mllvm -enable-mix=true)
            fi
            "$sdk/bin/ccec" -O2 -x cce --cce-auto-sync=off --cce-aicore-only \
                --cce-generic-addrspace=off --cce-aicore-arch="dav-c220-$arch" \
                -std=c++17 -I"$asc/include" -I"$asc/include/interface" \
                -I"$asc/include/basic_api" -I"$asc" -I"$sdk/aarch64-linux/ascendc/include/highlevel_api" \
                "${flags[@]}" "$stem.cpp" -c -emit-llvm -o "$build/$stem.$mode.bc"
        done
        "$sdk/bin/llvm-link" "$build/$stem.normal.bc" --override="$build/$stem.mixed.bc" -o "$stem.bc"
    done
done
