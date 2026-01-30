# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FAST-LIVO2 is a LiDAR-Inertial-Visual Odometry (LIVO) system built as a ROS package. It performs real-time 3D reconstruction and localization by fusing LiDAR, IMU, and camera data.

## Build System and Commands

### Prerequisites
- ROS (Melodic/Noetic)
- PCL >= 1.8
- Eigen >= 3.3.4
- OpenCV >= 4.2
- Sophus (specific commit: a621ff)
- Vikit (from xuankuzcr/rpg_vikit.git)

### Build Commands
```bash
# From catkin workspace root
catkin_make
source ~/catkin_ws/devel/setup.bash
```

### Run Commands
```bash
# Launch with Livox AVIA LiDAR (most common)
roslaunch fast_livo mapping_avia.launch

# Alternative launch files for different sensors:
roslaunch fast_livo mapping_hesaixt32_hilti22.launch  # HILTI22 dataset
roslaunch fast_livo mapping_ouster_ntu.launch        # NTU dataset
roslaunch fast_livo mapping_avia_marslvig.launch     # MARS-LVIG dataset

# Play rosbag data
rosbag play YOUR_DOWNLOADED.bag
```

## Architecture Overview

### Core Components

1. **LIVMapper** (`src/LIVMapper.cpp`, `include/LIVMapper.h`)
   - Main system orchestrator that coordinates all subsystems
   - Handles sensor data synchronization and state estimation
   - Manages ROS publishers/subscribers and system lifecycle

2. **VIO Manager** (`src/vio.cpp`, `include/vio.h`)
   - Visual-inertial odometry processing
   - Feature tracking and visual SLAM
   - Camera-IMU fusion

3. **LIO Processing** (`src/voxel_map.cpp`, `include/voxel_map.h`)
   - LiDAR-inertial odometry 
   - Voxel-based mapping structure
   - Point cloud registration

4. **IMU Processing** (`src/IMU_Processing.cpp`)
   - IMU data preprocessing and integration
   - Bias estimation and compensation

5. **Preprocessing** (`src/preprocess.cpp`, `include/preprocess.h`)
   - LiDAR point cloud filtering and organization
   - Supports multiple LiDAR types (Livox, Ouster, Hesai, Velodyne)

### Data Flow
1. Raw sensor data (LiDAR, IMU, camera) received via ROS topics
2. Preprocessing filters and organizes point clouds
3. IMU data integrated for motion prediction
4. VIO processes visual features and camera poses
5. LIO performs scan matching and mapping
6. Final pose estimation combines VIO and LIO results

### Configuration System

Configuration files in `config/` directory control system behavior:
- `avia.yaml`: Main system parameters for Livox AVIA
- `camera_*.yaml`: Camera-specific calibration parameters
- `HILTI22.yaml`, `NTU_VIRAL.yaml`: Dataset-specific configurations

Key configuration sections:
- `common`: Topic names and sensor enable flags
- `extrin_calib`: Extrinsic calibration between sensors
- `vio`: Visual odometry parameters
- `lio`: LiDAR odometry parameters
- `imu`: IMU noise and bias parameters

### Launch System

Launch files in `launch/` specify:
- Parameter loading from YAML files
- Node execution with proper arguments
- RViz visualization setup
- Image transport configuration

## Development Notes

- Built with C++17 standard
- Uses architecture-specific optimizations (ARM/x86)
- Supports multithreading with OpenMP when available
- Optional mimalloc integration for memory optimization
- Debug builds include gdb and valgrind integration options