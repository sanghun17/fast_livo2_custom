#!/bin/bash
# FAST-LIVO runtime selected by FAST_LIVO_PROFILE=hardware|airsim.
set -e

if [ ! -f /.dockerenv ]; then
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/scripts/lib/select_stack.sh"
  dsd_select_stack "odometry/fast-livo" || exit $?
  : "${DSD_CONTAINER:?set DSD_CONTAINER or invoke through './setup.sh run <stack> odometry/fast-livo'}"
  __C="$DSD_CONTAINER"
  __S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  __R="$(cd "$(dirname "$__S")/../../.." && pwd)"
  source "$__R/scripts/lib/ensure_container.sh"
  docker start "$__C" >/dev/null 2>&1
  __PROFILE="$(docker exec "$__C" printenv FAST_LIVO_PROFILE 2>/dev/null || true)"
  case "${__PROFILE:-hardware}" in
    hardware) __M="roslaunch fast_livo mapping_d435i.launch" ;;
    airsim) __M="roslaunch fast_livo mapping_simulator_openvins.launch" ;;
    *) echo "ERROR: invalid FAST_LIVO_PROFILE=$__PROFILE" >&2; exit 2 ;;
  esac
  __TT=$([ -t 1 ] && echo -it || echo -i)
  cleanup(){ docker exec "$__C" pkill -INT -f "$__M" >/dev/null 2>&1 || true; }
  trap 'cleanup; exit 130' INT TERM HUP
  docker exec $__TT "$__C" bash "/work/${__S#$__R/}" "$@"; __rc=$?
  cleanup
  exit $__rc
fi

source /opt/ros/noetic/setup.bash
case "${FAST_LIVO_PROFILE:-hardware}" in
  hardware)
    source /work/ws/fast-livo/devel/setup.bash
    source /work/config/ros_env.sh
    source /work/scripts/lib/ensure_roscore.sh
    exec taskset -c "${CPUS_POOL:?config/ros_env.sh not sourced}" \
      roslaunch fast_livo mapping_d435i.launch "$@"
    ;;
  airsim)
    source /work/ws/fast-livo-sim/devel/setup.bash
    source /work/config/sim.env
    source /work/config/ros_env.sh
    source /work/scripts/lib/ensure_roscore.sh
    exec roslaunch fast_livo mapping_simulator_openvins.launch "$@"
    ;;
  *)
    echo "ERROR: FAST_LIVO_PROFILE must be hardware or airsim" >&2
    exit 2
    ;;
esac
