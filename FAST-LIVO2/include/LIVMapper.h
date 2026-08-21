/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIV_MAPPER_H
#define LIV_MAPPER_H

#include "IMU_Processing.h"
#include "imu_init_buffer.h"
#include "vio.h"
#include "preprocess.h"
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Path.h>
#include <vikit/camera_loader.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Transform.h>
#include <std_msgs/Empty.h>
#include <cstdint>
#include <limits>
#include <memory>

class LIVMapper
{
public:
  LIVMapper(ros::NodeHandle &nh);
  ~LIVMapper();
  void initializeSubscribersAndPublishers(ros::NodeHandle &nh);
  void initializeComponents();
  void initializeFiles();
  void run();
  void gravityAlignment();
  void handleFirstFrame();
  void stateEstimationAndMapping();
  void handleVIO();
  void handleLIO();
  void savePCD();
  void processImu();
  void maybeLatchLiveImuInitAnchor(double synchronized_epoch,
                                   double lidar_watermark,
                                   double image_epoch);
  void failImuInitialization(const std::string &reason,
                             std::uint64_t synchronized_epoch_ns = 0);
  void logAcceptedImuInitialization(std::uint64_t state_epoch_ns);
  
  bool sync_packages(LidarMeasureGroup &meas);
  void prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr);
  void enqueue_imu_correction(const StatesGroup &state, double stamp);
  void publish_imu_propagated(const sensor_msgs::Imu &imu);
  void publish_correction_pose_cov(const StatesGroup &state, double stamp);
  void imu_prop_callback(const ros::TimerEvent &e);
  void transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud);
  void pointBodyToWorld(const PointType &pi, PointType &po);
 
  void RGBpointBodyToWorld(PointType const *const pi, PointType *const po);
  void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in);
  void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in);
  void img_cbk(const sensor_msgs::ImageConstPtr &msg_in);
  // OptiTrack/mocap gt-init (ported from the sim `ml` branch): latch odom->camera_init
  // from the first mocap pose so the published odom-frame topics start at the true pose.
  void gt_odom_cbk(const geometry_msgs::PoseStamped::ConstPtr &msg_in);
  // re-anchor on demand (std_msgs/Empty trigger): reset LIVO state, re-latch next pose.
  void reinit_cbk(const std_msgs::Empty::ConstPtr &msg_in);
  void publish_img_rgb(const ros::Publisher &pubImage, VIOManagerPtr vio_manager);
  void publish_frame_world(const ros::Publisher &pubLaserCloudFullRes, VIOManagerPtr vio_manager);
  void publish_visual_sub_map(const ros::Publisher &pubSubVisualMap);
  void publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list);
  void publish_odometry(const ros::Publisher &pubOdomAftMapped);
  void publish_body_optitrack();  // /aft_mapped_to_body (+ /aft_mapped_to_optitrack) via hand-eye T_cam2body
  void publish_path(const ros::Publisher pubPath);
  ros::Time estimator_stamp() const;
  void readParameters(ros::NodeHandle &nh);
  template <typename T> void set_posestamp(T &out);
  template <typename T> void pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi, Eigen::Matrix<T, 3, 1> &po);
  template <typename T> Eigen::Matrix<T, 3, 1> pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi);
  cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);

  std::mutex mtx_buffer, mtx_buffer_imu_prop;
  std::condition_variable sig_buffer;

  SLAM_MODE slam_mode_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map;
  
  string root_dir;
  string lid_topic, imu_topic, seq_name, img_topic;
  // Online camera intrinsics: grab one CameraInfo at init; the static yaml named by
  // cam_calib_file is loaded on demand only when the topic is absent.
  string cam_info_topic;
  string cam_calib_file;
  bool online_intrinsics_en = false;
  double cam_info_timeout = 5.0;
  V3D extT;
  M3D extR;

  // Body(=marker) hand-eye extrinsic. /aft_mapped_to_body (PoseStamped, W_L frame, mux vio input)
  // + /aft_mapped_to_optitrack (Odometry, GT-anchored). T_cam2body (p_body=R*p_cam+t) from config body_calib.
  bool body_calib_en = false;
  M3D R_cam2body = M3D::Identity();
  V3D t_cam2body = V3D::Zero();
  double gt_avg_sec = 3.0;
  bool mocap_anchor_enable = true;
  bool runtime_reinit_enable = false;
  bool body_anchor_latched = false;
  Eigen::Isometry3d T_anchor = Eigen::Isometry3d::Identity();   // optitrack <- W_L, latched once
  std::vector<std::pair<double, Eigen::Isometry3d>> gt_buf;     // recent GT (optitrack<-body) for the anchor avg

  int feats_down_size = 0, max_iterations = 0;

  double res_mean_last = 0.05;
  double gyr_cov = 0, acc_cov = 0, inv_expo_cov = 0;
  double b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
  double imu_init_max_gyr_mean = 0.30;
  double imu_init_max_gyr_std = 0.25;
  double imu_init_max_acc_std = 1.50;
  double imu_init_acc_norm_tolerance = 3.00;
  double blind_rgb_points = 0.0;
  double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0, last_timestamp_img = -1.0;
  double filter_size_surf_min = 0;
  double filter_size_pcd = 0;
  double _first_lidar_time = 0.0;
  double match_time = 0, solve_time = 0, solve_const_H_time = 0;

  bool lidar_map_inited = false, pcd_save_en = false, pub_effect_point_en = false, pose_output_en = false, ros_driver_fix_en = false, hilti_en = false;
  int pcd_save_interval = -1, pcd_index = 0;
  int pub_scan_num = 1;

  struct ImuCorrectionSnapshot
  {
    double stamp;
    StatesGroup state;
  };
  StatesGroup imu_propagate;

  bool imu_prop_enable = true, ekf_finish_once = false;
  bool imu_propagation_valid = false;
  deque<sensor_msgs::Imu> prop_imu_buffer;
  deque<sensor_msgs::Imu> prop_imu_history;
  deque<ImuCorrectionSnapshot> prop_correction_buffer;
  double last_propagated_imu_stamp = std::numeric_limits<double>::quiet_NaN();
  double last_prop_imu_input_stamp = std::numeric_limits<double>::quiet_NaN();
  double last_prop_correction_stamp = std::numeric_limits<double>::quiet_NaN();
  double imu_prop_max_dt = 0.05;
  int imu_prop_queue_max = 4096;
  int imu_prop_correction_queue_max = 256;
  std::uint64_t imu_prop_invalid_sample_count = 0;
  std::uint64_t imu_prop_nonmonotonic_count = 0;
  std::uint64_t imu_prop_gap_count = 0;
  std::uint64_t imu_prop_queue_drop_count = 0;
  std::uint64_t imu_prop_history_drop_count = 0;
  std::uint64_t imu_prop_correction_drop_count = 0;
  std::uint64_t imu_prop_superseded_count = 0;
  std::size_t imu_prop_pending_high_water = 0;
  std::size_t imu_prop_history_high_water = 0;
  std::size_t imu_prop_correction_high_water = 0;
  nav_msgs::Odometry imu_prop_odom;
  ros::Publisher pubImuPropOdom;
  ros::Publisher pubImuPropWorldTwist;
  ros::Publisher pubCorrectionPoseCov;
  double imu_time_offset = 0.0;
  double lidar_time_offset = 0.0;

  bool gravity_align_en = false, gravity_align_finished = false, imu_only_mode = false, fusion_debug = false, vio_flip_roll = false, vio_flip_pitch = false;
  bool visual_quality_log = false;
  std::string visual_quality_output_prefix = "/tmp/fast_livo_visual_quality";
  int visual_quality_flush_every_n_frames = 10;
  int vio_max_lio_features_for_fusion = -1;
  FILE *dbg_fp = nullptr;   // debug-only per-frame fusion/preintegration log (debug/fusion_log)

  bool sync_jump_flag = false;

  bool lidar_pushed = false, imu_en, gravity_est_en, flg_reset = false, ba_bg_est_en = true;
  bool imu_init_estimate_gyr_bias = false;
  bool dense_map_en = false;
  int img_en = 1, imu_int_frame = 3;
  bool normal_en = true;
  bool exposure_estimate_en = false;
  double exposure_time_init = 0.0;
  bool inverse_composition_en = false;
  bool raycast_en = false;
  int lidar_en = 1;
  bool is_first_frame = false;
  int grid_size, patch_size, grid_n_width, grid_n_height, patch_pyrimid_level;
  double outlier_threshold;
  double plot_time;
  int frame_cnt;
  bool verbose = false;   // debug/verbose: gate per-frame VIO/LIO console spam (rosparam, set in launch, no rebuild)
  double img_time_offset = 0.0;
  deque<PointCloudXYZI::Ptr> lid_raw_data_buffer;
  deque<double> lid_header_time_buffer;
  deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
  using ImuInitSampleBuffer =
      fast_livo::ImuInitBuffer<sensor_msgs::Imu::ConstPtr>;
  std::unique_ptr<ImuInitSampleBuffer> imu_init_buffer;
  std::string imu_init_anchor_stamp_ns_param;
  int imu_init_queue_max = 4096;
  double imu_init_anchor_max_predecessor_gap_s = 0.02;
  bool imu_init_anchor_explicit = false;
  bool imu_init_failed = false;
  bool imu_init_accepted_logged = false;
  bool imu_init_first_correction_logged = false;
  std::string imu_init_initial_state_sha256;
  std::string imu_init_failure_reason;
  std::uint64_t imu_init_anchor_lidar_watermark_ns = 0;
  std::uint64_t imu_init_anchor_image_epoch_ns = 0;
  std::uint64_t imu_init_anchor_imu_watermark_ns = 0;
  std::uint64_t imu_init_last_sync_epoch_ns = 0;
  int imu_input_queue_max = 4096;
  std::uint64_t imu_input_queue_drop_count = 0;
  std::uint64_t imu_pre_lidar_sample_count = 0;
  std::size_t imu_input_high_water = 0;
  deque<cv::Mat> img_buffer;
  deque<double> img_time_buffer;
  int img_input_queue_max = 64;
  std::uint64_t img_input_queue_drop_count = 0;
  std::uint64_t img_pre_lidar_frame_count = 0;
  std::size_t img_input_high_water = 0;
  vector<pointWithVar> _pv_list;
  vector<double> extrinT;
  vector<double> extrinR;
  vector<double> cameraextrinT;
  vector<double> cameraextrinR;
  double IMG_POINT_COV;

  PointCloudXYZI::Ptr visual_sub_map;
  PointCloudXYZI::Ptr feats_undistort;
  PointCloudXYZI::Ptr feats_down_body;
  PointCloudXYZI::Ptr feats_down_world;
  PointCloudXYZI::Ptr pcl_w_wait_pub;
  PointCloudXYZI::Ptr pcl_wait_pub;
  PointCloudXYZRGB::Ptr pcl_wait_save;
  PointCloudXYZI::Ptr pcl_wait_save_intensity;

  ofstream fout_pre, fout_out, fout_pcd_pos, fout_points;

  pcl::VoxelGrid<PointType> downSizeFilterSurf;

  V3D euler_cur;

  LidarMeasureGroup LidarMeasures;
  StatesGroup _state;
  StatesGroup  state_propagat;

  nav_msgs::Path path;
  nav_msgs::Odometry odomAftMapped;
  geometry_msgs::Quaternion geoQuat;
  geometry_msgs::PoseStamped msg_body_pose;

  PreprocessPtr p_pre;
  ImuProcessPtr p_imu;
  VoxelMapManagerPtr voxelmap_manager;
  VIOManagerPtr vio_manager;

  ros::Publisher plane_pub;
  ros::Publisher voxel_pub;
  ros::Subscriber sub_pcl;
  ros::Subscriber sub_imu;
  ros::Subscriber sub_img;
  ros::Subscriber sub_gt_odom;   // mocap pose -> gt_odom_cbk (always subscribed; self-selects)
  ros::Subscriber sub_reinit;    // /livo/reinit -> reinit_cbk
  // mocap gt-init state. odom_to_camera_init holds the latched odom<-camera_init pose;
  // the imu_prop odom path multiplies by it (gt_odom_cbk latches it from the first VRPN pose).
  bool gt_odom_received = false;
  std::string gt_pose_topic;
  geometry_msgs::Transform odom_to_camera_init;
  ros::Publisher pubLaserCloudFullRes;
  ros::Publisher pubNormal;
  ros::Publisher pubSubVisualMap;
  ros::Publisher pubLaserCloudEffect;
  ros::Publisher pubLaserCloudMap;
  ros::Publisher pubOdomAftMapped;
  ros::Publisher pubOdomAftMappedOdom;  // Odometry in odom frame
  ros::Publisher pubOdomAftMappedBody;  // /aft_mapped_to_body (PoseStamped, W_L, mux vio input)
  ros::Publisher pubPath;
  ros::Publisher pubLaserCloudDyn;
  ros::Publisher pubLaserCloudDynRmed;
  ros::Publisher pubLaserCloudDynDbg;
  ros::Publisher pubImage;
  ros::Timer imu_prop_timer;

  int frame_num = 0;
  double aver_time_consu = 0;
  double aver_time_icp = 0;
  double aver_time_map_inre = 0;
  bool colmap_output_en = false;
};
#endif
