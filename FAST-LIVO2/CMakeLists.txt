cmake_minimum_required(VERSION 2.8.3)
project(fast_livo)

set(CMAKE_BUILD_TYPE "Release")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set common compile options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions")

# Specific settings for Debug build
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

# Detect CPU architecture
message(STATUS "Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")

# Specific settings for Release build
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64|ARM|AARCH64)")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # 64-bit ARM optimizations (e.g., RK3588 and Jetson Orin NX)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native")
    message(STATUS "Using 64-bit ARM optimizations: -O3 -mcpu=native -mtune=native")
  else()
    # 32-bit ARM optimizations with NEON support
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native -mfpu=neon")
    message(STATUS "Using 32-bit ARM optimizations: -O3 -mcpu=native -mtune=native -mfpu=neon")
  endif()
  add_definitions(-DARM_ARCH)
else()
  # x86-64 (Intel/AMD) optimizations
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -funroll-loops") #-flto
  message(STATUS "Using general x86 optimizations: -O3 -march=native -mtune=native -funroll-loops") 
  add_definitions(-DX86_ARCH)
endif()

# Define project root directory
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

# Detect CPU core count for diagnostics.  Estimator concurrency is deliberately
# opt-in: a single-threaded default is slower, but reproducible and avoids
# silently changing numerical behavior between build hosts.
include(ProcessorCount)
ProcessorCount(N)
message(STATUS "Processor count: ${N}")

set(FASTLIVO_OPENMP_THREADS "1" CACHE STRING
    "FAST-LIVO OpenMP worker count (1 disables estimator parallel regions)")
if(NOT FASTLIVO_OPENMP_THREADS MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR
          "FASTLIVO_OPENMP_THREADS must be a positive integer, got '${FASTLIVO_OPENMP_THREADS}'")
endif()

# Check for OpenMP support
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found")
  add_compile_options(${OpenMP_CXX_FLAGS})
  if(FASTLIVO_OPENMP_THREADS GREATER 1)
    add_definitions(-DMP_EN -DMP_PROC_NUM=${FASTLIVO_OPENMP_THREADS})
    message(STATUS "FAST-LIVO OpenMP estimator threads: ${FASTLIVO_OPENMP_THREADS}")
  else()
    # Merely omitting our explicit OpenMP regions is not enough: Eigen detects
    # -fopenmp and may parallelize large H' * H / H' * z products internally.
    # Their reduction order is scheduler-dependent and can change an iterative
    # EKF accept/rollback branch between identical replays.  Make the default
    # genuinely single-threaded at compile time.
    add_definitions(-DMP_PROC_NUM=1 -DEIGEN_DONT_PARALLELIZE)
    message(STATUS "FAST-LIVO and Eigen estimator parallel regions disabled (safe default)")
  endif()
else()
  message(STATUS "OpenMP not found, proceeding without it")
  add_definitions(-DMP_PROC_NUM=1 -DEIGEN_DONT_PARALLELIZE)
endif()

# Check for mimalloc support
find_package(mimalloc QUIET)
if(mimalloc_FOUND)
  message(STATUS "mimalloc found")
else()
  message(STATUS "mimalloc not found, proceeding without it")
endif()

# Find catkin and required dependencies
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  message_generation
  eigen_conversions
  vikit_common
  vikit_ros
  cv_bridge
  image_transport
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)
find_package(OpenSSL REQUIRED)

set(Sophus_LIBRARIES libSophus.so)

# Define the catkin package
catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime cv_bridge vikit_common vikit_ros image_transport
  DEPENDS EIGEN3 PCL OpenCV Sophus
)

# Include directories for dependencies
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  include
)

# Add libraries
add_library(vio src/vio.cpp src/frame.cpp src/visual_point.cpp)
add_library(lio src/voxel_map.cpp)
add_library(pre src/preprocess.cpp)
add_library(imu_proc src/IMU_Processing.cpp)
add_library(laser_mapping src/LIVMapper.cpp)
if(OpenMP_CXX_FOUND)
  target_link_libraries(imu_proc PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(laser_mapping OpenSSL::Crypto)

# Add the main executable
add_executable(fastlivo_mapping src/main.cpp)

if(CATKIN_ENABLE_TESTING)
  add_executable(test_imu_init_buffer test/test_imu_init_buffer.cpp)
  add_test(NAME fast_livo_imu_init_buffer COMMAND test_imu_init_buffer)
  add_executable(test_imu_initialization test/test_imu_initialization.cpp)
  target_link_libraries(test_imu_initialization
    imu_proc
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
  )
  add_test(NAME fast_livo_imu_initialization COMMAND test_imu_initialization)
endif()

# Link libraries to the executable
target_link_libraries(fastlivo_mapping
  laser_mapping
  vio
  lio
  pre
  imu_proc
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Sophus_LIBRARIES}
  ${Boost_LIBRARIES}
)

# Link mimalloc if found
if(mimalloc_FOUND)
  target_link_libraries(fastlivo_mapping mimalloc)
endif()
