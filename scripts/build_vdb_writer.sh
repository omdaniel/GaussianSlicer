#!/usr/bin/env bash
set -euo pipefail

# Build the minimal OpenVDB writer using Homebrew + pkg-config
# Requirements: brew install openvdb pkg-config

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
OUT="$ROOT/Tools/vdb_writer"

# Ensure Homebrew bin is on PATH (Apple Silicon default)
if command -v brew >/dev/null 2>&1; then
  eval "$(brew shellenv)"
fi

# Seed PKG_CONFIG_PATH with common Homebrew locations
BREW_PREFIX=${HOMEBREW_PREFIX:-${HOMEBREW_PREFIX:-/opt/homebrew}}
PC_PATHS=(
  "$BREW_PREFIX/lib/pkgconfig"
  "$BREW_PREFIX/opt/openvdb/lib/pkgconfig"
  "$BREW_PREFIX/opt/imath/lib/pkgconfig"
  "$BREW_PREFIX/opt/openexr/lib/pkgconfig"
  "$BREW_PREFIX/opt/tbb/lib/pkgconfig"
)
for p in "${PC_PATHS[@]}"; do
  if [ -d "$p" ]; then
    export PKG_CONFIG_PATH="$p:${PKG_CONFIG_PATH:-}"
  fi
done

if ! command -v pkg-config >/dev/null 2>&1; then
  echo "pkg-config is required. brew install pkg-config" >&2
  exit 1
fi

mkdir -p "$ROOT/Tools"

if pkg-config --exists openvdb; then
  CXXFLAGS="$(pkg-config --cflags openvdb) -std=c++17 -O2"
  LDFLAGS="$(pkg-config --libs openvdb)"
  echo "Using pkg-config detected flags"
  echo "CXXFLAGS: $CXXFLAGS"
  echo "LDFLAGS: $LDFLAGS"
  echo "Building vdb_writer -> $OUT"
  clang++ $CXXFLAGS "$ROOT/Tools/vdb_writer.cpp" -o "$OUT" $LDFLAGS
else
  echo "openvdb.pc not found; falling back to Homebrew prefixes" >&2
  ODB_PREFIX="$(brew --prefix openvdb)"
  TBB_PREFIX="$(brew --prefix tbb)"
  BLOSC_PREFIX="$(brew --prefix c-blosc)"
  BOOST_PREFIX="$(brew --prefix boost)"
  INC_FLAGS=(
    -I"$ODB_PREFIX/include"
    -I"$BREW_PREFIX/include"
  )
  LIB_FLAGS=(
    -L"$ODB_PREFIX/lib"
    -L"$BREW_PREFIX/lib"
    -lopenvdb -ltbb -lblosc -lboost_iostreams -lboost_random -lboost_regex -lz
  )
  SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
  CPPFLAGS=( -isysroot "$SDKROOT" )
  # Ensure libc++ headers are found (new CLT places them under the SDK)
  if [ -d "$SDKROOT/usr/include/c++/v1" ]; then
    CPPFLAGS+=( -I"$SDKROOT/usr/include/c++/v1" )
  fi
  CXXSTD=( -std=c++17 -O2 -stdlib=libc++ )
  echo "SDKROOT: $SDKROOT"
  echo "INC: ${INC_FLAGS[*]}"
  echo "LIB: ${LIB_FLAGS[*]}"
  echo "Building vdb_writer -> $OUT"
  clang++ "${CXXSTD[@]}" "${CPPFLAGS[@]}" "${INC_FLAGS[@]}" "$ROOT/Tools/vdb_writer.cpp" -o "$OUT" "${LIB_FLAGS[@]}"
fi

chmod +x "$OUT"
echo "Done."
