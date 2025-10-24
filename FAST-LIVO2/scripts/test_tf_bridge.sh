#!/bin/bash
# Test script for smart_static_tf_bridge.py
#
# This script creates a test scenario to verify the TF bridge works correctly

echo "========================================"
echo "Testing Smart Static TF Bridge"
echo "========================================"
echo ""

# Step 1: Check prerequisites
echo "Step 1: Checking prerequisites..."
echo ""

if ! rostopic list &>/dev/null; then
    echo "ERROR: ROS master is not running!"
    echo "Please run: roscore"
    exit 1
fi

echo "✓ ROS master is running"
echo ""

# Step 2: Check if RealSense is publishing TF
echo "Step 2: Checking RealSense TF..."
if timeout 2 rostopic echo /tf_static -n 1 | grep -q "camera_link"; then
    echo "✓ RealSense is publishing TF"
else
    echo "WARNING: RealSense TF not detected"
    echo "  Make sure RealSense camera is running"
fi
echo ""

# Step 3: Create a fake 'aft_mapped' frame for testing
echo "Step 3: Creating fake 'aft_mapped' frame for testing..."
echo "  Publishing: camera_init -> aft_mapped (static identity)"

# Start fake SLAM TF in background
rosrun tf static_transform_publisher 0 0 0 0 0 0 camera_init aft_mapped 100 &
FAKE_SLAM_PID=$!

sleep 2

echo "✓ Fake SLAM TF published (PID: $FAKE_SLAM_PID)"
echo ""

# Step 4: List available frames
echo "Step 4: Checking available frames..."
echo "  camera_link (RealSense root) - should exist"
echo "  aft_mapped (SLAM pose) - just created"
echo "  camera_color_optical_frame (RealSense optical) - should exist"
echo ""

# Step 5: Run the smart TF bridge
echo "Step 5: Running smart TF bridge..."
echo "========================================"
echo ""

# Run the bridge (it will exit after publishing)
rosrun fast_livo smart_static_tf_bridge.py \
    _parent:=aft_mapped \
    _child:=camera_color_optical_frame \
    _x:=0 _y:=0 _z:=0 \
    _qx:=0 _qy:=0 _qz:=0 _qw:=1

BRIDGE_EXIT_CODE=$?

echo ""
echo "========================================"
echo "Step 6: Verification"
echo "========================================"
echo ""

if [ $BRIDGE_EXIT_CODE -eq 0 ]; then
    echo "✓ Bridge exited successfully"
    echo ""

    # Check if transform was published
    echo "Checking published transform..."
    sleep 1

    if timeout 2 rostopic echo /tf_static -n 50 | grep -A 1 "frame_id: \"aft_mapped\"" | grep -q "child_frame_id: \"camera_link\""; then
        echo "✓ Transform published: aft_mapped -> camera_link"

        # Show the transform
        echo ""
        echo "Published transform:"
        timeout 2 rostopic echo /tf_static -n 50 | grep -A 15 "frame_id: \"aft_mapped\"" | head -20

    else
        echo "WARNING: Could not verify transform on /tf_static"
    fi

else
    echo "ERROR: Bridge exited with code $BRIDGE_EXIT_CODE"
fi

echo ""
echo "========================================"
echo "Cleanup"
echo "========================================"

# Kill fake SLAM
kill $FAKE_SLAM_PID 2>/dev/null
echo "✓ Stopped fake SLAM publisher"

echo ""
echo "Test complete!"
