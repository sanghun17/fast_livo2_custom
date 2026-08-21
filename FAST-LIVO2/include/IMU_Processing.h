/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef IMU_PROCESSING_H
#define IMU_PROCESSING_H

#include <Eigen/Eigen>
#include "common_lib.h"
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <nav_msgs/Odometry.h>
#include <utils/so3_math.h>
#include <fstream>
#include <utility>
const bool time_list(PointType &x, PointType &y) { return (x.curvature < y.curvature); }

/// *************IMU Process and undistortion
class ImuProcess
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();

  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4, 4) & T);
  void set_gyr_cov_scale(const V3D &scaler);
  void set_acc_cov_scale(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  void set_inv_expo_cov(const double &inv_expo);
  void set_imu_init_frame_num(const int &num);
  void set_imu_init_stationarity(const double max_gyr_mean,
                                 const double max_gyr_std,
                                 const double max_acc_std,
                                 const double acc_norm_tolerance,
                                 const bool estimate_gyr_bias);
  void disable_imu();
  void disable_gravity_est();
  void disable_bias_est();
  void disable_exposure_est();
  void Process2(
      LidarMeasureGroup &lidar_meas, StatesGroup &stat,
      PointCloudXYZI::Ptr cur_pcl_un_,
      const deque<sensor_msgs::Imu::ConstPtr> *init_imu_samples = nullptr);
  void UndistortPcl(LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out);

  // Read-only startup diagnostics used by deterministic replay tests.
  int imu_init_sample_count() const { return init_sample_count; }
  int imu_init_target_count() const { return MAX_INI_COUNT; }
  int imu_init_rejected_windows() const { return init_rejected_windows; }
  int imu_init_invalid_samples() const { return init_invalid_samples; }
  double imu_init_first_stamp() const { return init_first_stamp; }
  double imu_init_last_stamp() const { return init_last_stamp; }
  std::uint32_t imu_init_first_seq() const { return init_first_seq; }
  std::uint32_t imu_init_last_seq() const { return init_last_seq; }
  const V3D &imu_init_mean_acc() const { return mean_acc; }
  const V3D &imu_init_mean_gyr() const { return mean_gyr; }
  const vector<std::pair<std::uint64_t, std::uint32_t>> &
  imu_init_selected_samples() const { return init_selected_samples; }
  double imu_last_prop_end_time() const { return last_prop_end_time; }
  std::uint64_t imu_last_sample_stamp_ns() const
  {
    return last_imu ? last_imu->header.stamp.toNSec() : 0;
  }

  ofstream fout_imu;
  double IMU_mean_acc_norm;
  V3D unbiased_gyr;

  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  double cov_inv_expo;
  double first_lidar_time;
  bool imu_time_init = false;
  bool imu_need_init = true;
  M3D Eye3d;
  V3D Zero3d;
  int lidar_type;

private:
  bool IMU_init(const MeasureGroup &meas, StatesGroup &state);
  void reset_imu_init_window();
  void Forward_without_imu(LidarMeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out);
  PointCloudXYZI pcl_wait_proc;
  sensor_msgs::ImuConstPtr last_imu;
  PointCloudXYZI::Ptr cur_pcl_un_;
  vector<Pose6D> IMUpose;
  M3D Lid_rot_to_IMU;
  V3D Lid_offset_to_IMU;
  V3D mean_acc;
  V3D mean_gyr;
  V3D init_acc_m2 = V3D::Zero();
  V3D init_gyr_m2 = V3D::Zero();
  V3D angvel_last;
  V3D acc_s_last;
  double init_first_stamp = std::numeric_limits<double>::quiet_NaN();
  double init_last_stamp = std::numeric_limits<double>::quiet_NaN();
  std::uint32_t init_first_seq = 0;
  std::uint32_t init_last_seq = 0;
  vector<std::pair<std::uint64_t, std::uint32_t>> init_selected_samples;
  double init_max_gyr_mean = 0.30;
  double init_max_gyr_std = 0.25;
  double init_max_acc_std = 1.50;
  double init_acc_norm_tolerance = 3.00;
  double last_prop_end_time = 0.0;
  double time_last_scan;
  int init_sample_count = 0, MAX_INI_COUNT = 20;
  int init_rejected_windows = 0;
  int init_invalid_samples = 0;
  bool init_estimate_gyr_bias = false;
  bool b_first_frame = true;
  bool imu_en = true;
  bool gravity_est_en = true;
  bool ba_bg_est_en = true;
  bool exposure_estimate_en = true;
};
typedef std::shared_ptr<ImuProcess> ImuProcessPtr;
#endif
