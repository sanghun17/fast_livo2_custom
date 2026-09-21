#!/bin/bash
# Build the selected FAST-LIVO implementation workspace.
set -e
source /opt/ros/noetic/setup.bash
case "${FAST_LIVO_PROFILE:-hardware}" in
  hardware) cd /work/ws/fast-livo ;;
  airsim) cd /work/ws/fast-livo-sim ;;
  *) echo "ERROR: invalid FAST_LIVO_PROFILE=${FAST_LIVO_PROFILE}" >&2; exit 2 ;;
esac
catkin config --extend /opt/ros/noetic --cmake-args -DCMAKE_BUILD_TYPE=Release >/dev/null
catkin build
echo ">> fast-livo (${FAST_LIVO_PROFILE:-hardware}) workspace built"
