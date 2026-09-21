#!/bin/bash
# Fetch the implementation selected by FAST_LIVO_PROFILE. Both profiles provide
# the same odometry capability but the upstream branches have different layouts.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPO="${FASTLIVO_REPO:-git@github.com:sanghun17/fast_livo2_custom.git}"
PROFILE="${FAST_LIVO_PROFILE:-hardware}"

case "$PROFILE" in
  hardware)
    DST="$ROOT/ws/fast-livo/src"
    BRANCH="${FASTLIVO_BRANCH:-jetson-orin-agx}"
    bash "$ROOT/scripts/lib/clone_repo.sh" "$DST" "$REPO" "$BRANCH"
    ;;
  airsim)
    SRC="$ROOT/ws/fast-livo-sim/src"
    BRANCH="${FASTLIVO_SIM_BRANCH:-ml}"
    VIKIT_REPO="${VIKIT_REPO:-https://github.com/xuankuzcr/rpg_vikit.git}"
    VIKIT_COMMIT="${VIKIT_COMMIT:-6c886c8}"
    mkdir -p "$SRC"
    bash "$ROOT/scripts/lib/clone_repo.sh" "$SRC/FAST-LIVO2" "$REPO" "$BRANCH"
    if [ -d "$SRC/rpg_vikit/.git" ]; then
      git -C "$SRC/rpg_vikit" fetch origin
      git -C "$SRC/rpg_vikit" checkout "$VIKIT_COMMIT"
    else
      git clone "$VIKIT_REPO" "$SRC/rpg_vikit"
      git -C "$SRC/rpg_vikit" checkout "$VIKIT_COMMIT"
    fi
    ;;
  *)
    echo "ERROR: FAST_LIVO_PROFILE must be hardware or airsim (got '$PROFILE')" >&2
    exit 2
    ;;
esac
