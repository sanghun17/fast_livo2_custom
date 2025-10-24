# FAST-LIVO2 for Jetson Orin AGX + RealSense D455

This repository contains FAST-LIVO2 (Fast LiDAR-Inertial-Visual Odometry) configured for **Jetson Orin AGX** with **Intel RealSense D455** camera and **Mid-360 LiDAR**.

## Important Notes

⚠️ **This is NOT for I.MX8 board** - This repository is specifically configured for Jetson Orin AGX platform.

⚠️ **Integrated with EPIC** - This FAST-LIVO2 setup works with the EPIC exploration planner. See: https://github.com/sanghun17/EPIC_poongsan/tree/jetson-orin-agx

## Hardware Requirements

- **Platform**: NVIDIA Jetson Orin AGX (64GB recommended)
- **Camera**: Intel RealSense D455
- **LiDAR**: Livox Mid-360
- **IMU**: Built-in IMU in Mid-360 (or external IMU)

## Software Dependencies

- ROS Noetic
- OpenCV 4
- PCL (Point Cloud Library)
- Eigen3
- Intel RealSense SDK 2.0
- Livox SDK2
- cv_bridge
- image_transport

## Repository Contents

This repository includes:
- **FAST-LIVO2**: Main LiDAR-Inertial-Visual odometry package
- **rpg_vikit**: Vision toolkit for feature tracking
- **vision_opencv**: OpenCV-ROS bridge packages
- **Custom configurations**: D455 camera parameters
- **Launch files**: Ready-to-use launch files for D455
- **TF Bridge**: Smart static transform bridge for EPIC integration

## Key Features

### 1. D455 Camera Integration
- Camera intrinsics configured for RealSense D455
- Depth and RGB stream synchronization
- Optimized for Jetson Orin hardware acceleration

### 2. EPIC Integration Ready
- Publishes `/aft_mapped_to_odom` (nav_msgs/Odometry)
- Publishes `/cloud_registered` (sensor_msgs/PointCloud2)
- Static TF bridge maintains `camera_link` → `camera_init` transform
- Frame: `camera_init` (map frame) and `aft_mapped` (body frame)

### 3. Custom TF Bridge
- `smart_static_tf_bridge.py`: Maintains static transform for EPIC
- Stays alive with `rospy.spin()` to prevent transform loss
- Automatically detects and publishes required transforms

## Installation

### 1. Install Dependencies

```bash
# RealSense SDK
sudo apt-get install ros-noetic-realsense2-camera
sudo apt-get install ros-noetic-realsense2-description

# Livox SDK2
cd ~
git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd Livox-SDK2
mkdir build && cd build
cmake .. && make -j$(nproc)
sudo make install

# Other ROS dependencies
sudo apt-get install ros-noetic-cv-bridge ros-noetic-image-transport
sudo apt-get install ros-noetic-pcl-ros ros-noetic-eigen-conversions
```

### 2. Clone and Build

```bash
cd ~/
mkdir -p fast_livo2_ws/src
cd fast_livo2_ws/src

# Clone this repository
git clone -b jetson-orin-agx <your-repo-url> .

cd ..
catkin build -DCMAKE_BUILD_TYPE=Release
```

## Configuration Files

### D455 Camera Config: `config/d455.yaml`

Located at: `FAST-LIVO2/config/d455.yaml`

**Camera Intrinsics**:
```yaml
cam_fx: 615.3680419921875
cam_fy: 615.239013671875
cam_cx: 321.8349914550781
cam_cy: 243.62889099121094
cam_width: 640
cam_height: 480
```

**Key Parameters**:
```yaml
# LiDAR Settings
lid_topic: "/livox/lidar"
imu_topic: "/livox/imu"

# Camera Settings
img0_topic: "/camera/color/image_raw"

# Odometry Output
pub_odom_tf: true  # Publish TF transforms
publish_odometry_without_downsample: true
```

### Launch File: `launch/mapping_d455.launch`

Launches FAST-LIVO2 with D455 configuration and static TF bridge.

## Running FAST-LIVO2

### Terminal 1: Start Hardware Drivers

```bash
# Start RealSense D455
roslaunch realsense2_camera rs_camera.launch \
  align_depth:=true \
  color_width:=640 \
  color_height:=480 \
  color_fps:=30

# Start Livox Mid-360 (in another terminal)
roslaunch livox_ros_driver2 msg_MID360.launch
```

### Terminal 2: Launch FAST-LIVO2

```bash
cd ~/fast_livo2_ws
source devel/setup.bash
roslaunch fast_livo mapping_d455.launch
```

### Verify Topics

```bash
# Check odometry is publishing
rostopic echo /aft_mapped_to_odom -n 1

# Check registered point cloud
rostopic echo /cloud_registered -n 1

# Check TF tree
rosrun tf view_frames
# Should show: camera_link -> camera_init -> aft_mapped
```

## Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/aft_mapped_to_odom` | nav_msgs/Odometry | Odometry in camera_init frame |
| `/cloud_registered` | sensor_msgs/PointCloud2 | Registered point cloud in camera_init frame |
| `/Laser_map` | sensor_msgs/PointCloud2 | Full LiDAR map |
| `/path` | nav_msgs/Path | Trajectory path |
| `/tf` | tf2_msgs/TFMessage | Transform tree |

## TF Frames

```
camera_link (D455 camera frame)
    ↓
camera_init (map/world frame from FAST-LIVO)
    ↓
aft_mapped (body frame - current robot pose)
```

## Integration with EPIC

This FAST-LIVO2 setup is designed to work seamlessly with EPIC exploration planner:

1. **Start FAST-LIVO2** (this repository):
```bash
roslaunch fast_livo mapping_d455.launch
```

2. **Start EPIC** (separate repository):
```bash
cd ~/epic_ws
source devel/setup.bash
roslaunch exploration_manager d455.launch
```

EPIC will subscribe to:
- `/aft_mapped_to_odom` for odometry
- `/cloud_registered` for point cloud input

See EPIC repository for details: https://github.com/sanghun17/EPIC_poongsan/tree/jetson-orin-agx

## Custom Scripts

### `scripts/smart_static_tf_bridge.py`

Maintains static transform between FAST-LIVO frames and camera frames:
- Waits for `/aft_mapped_to_odom` to be published
- Publishes static transform: `camera_link` → `camera_init`
- **Critical**: Uses `rospy.spin()` to stay alive (required for EPIC)

Usage:
```bash
rosrun fast_livo smart_static_tf_bridge.py
```

### `scripts/test_tf_bridge.sh`

Test script to verify TF bridge is working correctly.

## Troubleshooting

### Problem: No odometry published
**Cause**: FAST-LIVO not initialized or LiDAR/Camera not connected
**Solution**:
1. Check hardware connections: `rostopic list`
2. Verify LiDAR data: `rostopic echo /livox/lidar -n 1`
3. Verify camera images: `rostopic echo /camera/color/image_raw -n 1`
4. Check FAST-LIVO console for initialization messages

### Problem: TF tree disconnected
**Cause**: Static TF bridge not running or exited prematurely
**Solution**:
1. Verify bridge is running: `rosnode list | grep tf_bridge`
2. Restart bridge: `rosrun fast_livo smart_static_tf_bridge.py`
3. Check transform: `rosrun tf tf_echo camera_link camera_init`

### Problem: Poor odometry quality
**Cause**: Insufficient features or motion too fast
**Solution**:
1. Move slowly during initialization (first 5-10 seconds)
2. Ensure textured environment for visual features
3. Check camera focus and exposure settings
4. Reduce motion speed

### Problem: Point cloud not registered properly
**Cause**: Camera-LiDAR calibration or timing issues
**Solution**:
1. Verify camera intrinsics in `d455.yaml`
2. Check time synchronization between sensors
3. Adjust `time_sync_en` parameter if needed

## Performance Notes

### Jetson Orin AGX Settings
- Build in **Release mode**: `catkin build -DCMAKE_BUILD_TYPE=Release`
- Odometry rate: ~30-50 Hz
- Point cloud registration: ~10-20 Hz
- CPU usage: ~40-60% (4-6 cores)

### Memory Usage
- Typical RAM: 3-5 GB
- Peak with large maps: 8-12 GB
- Recommended: 32GB+ system RAM

### Power Settings
For optimal performance on Jetson:
```bash
# Max performance mode
sudo nvpmodel -m 0
sudo jetson_clocks
```

## Differences from Original FAST-LIVO2

### Changes Made:
1. **D455 Configuration**: Added camera intrinsics and parameters for RealSense D455
2. **Smart TF Bridge**: Modified to use `rospy.spin()` instead of timeout exit
3. **Launch Files**: Created D455-specific launch file
4. **EPIC Integration**: Added necessary topics and frames for EPIC compatibility
5. **Removed Submodules**: Integrated rpg_vikit and vision_opencv as direct packages

### Original Repository
Based on: https://github.com/hku-mars/FAST-LIVO2

## Repository Structure

```
fast_livo2_ws/src/
├── FAST-LIVO2/
│   ├── config/
│   │   ├── d455.yaml              # D455 configuration
│   │   └── camera_d455.yaml       # Camera parameters
│   ├── launch/
│   │   └── mapping_d455.launch    # Main launch file
│   ├── scripts/
│   │   ├── smart_static_tf_bridge.py  # TF bridge
│   │   └── test_tf_bridge.sh
│   ├── include/
│   ├── src/
│   └── CMakeLists.txt
├── rpg_vikit/                     # Vision toolkit
├── vision_opencv/                 # OpenCV bridge
└── README_JETSON_ORIN_D455.md    # This file
```

## Known Issues

1. **Initialization requires motion**: Need to move camera/robot slowly during first 5-10 seconds
2. **Visual degradation in low light**: D455 performance drops in dark environments
3. **LiDAR range limitations**: Mid-360 effective range ~20-30m outdoors

## Future Improvements

- [ ] Auto-exposure adjustment for varying lighting
- [ ] IMU bias online calibration
- [ ] Loop closure detection
- [ ] Map save/load functionality

## Credits

- **Original FAST-LIVO2**: [HKU-MARS FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2)
- **rpg_vikit**: [RPG Vision Toolkit](https://github.com/uzh-rpg/rpg_vikit)
- **vision_opencv**: [ROS OpenCV Bridge](https://github.com/ros-perception/vision_opencv)
- **This Integration**: Configured for Jetson Orin AGX with RealSense D455

## Citation

If you use this work, please cite the original FAST-LIVO2 paper:

```bibtex
@article{zheng2022fast,
  title={FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry},
  author={Zheng, Chunran and Zhu, Qingyan and Xu, Wei and Liu, Xiyuan and Guo, Qizhi and Zhang, Fu},
  journal={arXiv preprint arXiv:2209.01540},
  year={2022}
}
```

## License

Same as original FAST-LIVO2 project (check individual package licenses).

## Contact

For issues specific to this Jetson Orin + D455 integration, please open an issue on this repository's `jetson-orin-agx` branch.
