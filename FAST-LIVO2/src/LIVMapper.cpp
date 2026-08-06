/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "LIVMapper.h"
#include <sensor_msgs/CameraInfo.h>
#include <ros/topic.h>
#include <cstdlib>
#include <cmath>

namespace {
// Pick the vikit camera model from a distortion-model string and build it.
// radtan/plumb_bob (and empty) -> PinholeCamera; fisheye/equidistant -> EquidistantCamera.
vk::AbstractCamera *makeCamera(const std::string &dist_model, double w, double h, double scale,
                               double fx, double fy, double cx, double cy,
                               double d0, double d1, double d2, double d3)
{
  if (dist_model == "equidistant" || dist_model == "fisheye" || dist_model == "EquidistantCamera")
    return new vk::EquidistantCamera(w, h, scale, fx, fy, cx, cy, d0, d1, d2, d3);
  return new vk::PinholeCamera(w, h, scale, fx, fy, cx, cy, d0, d1, d2, d3);
}

// Online intrinsics: build the camera straight from a CameraInfo message.
// K = [fx 0 cx; 0 fy cy; 0 0 1]; D = distortion coeffs. `scale` is FAST-LIVO's
// image-downsample factor (not in CameraInfo), passed through.
bool buildCameraFromInfo(const sensor_msgs::CameraInfo &ci, double scale, vk::AbstractCamera *&cam)
{
  if (ci.width == 0 || ci.height == 0 || ci.K[0] == 0.0)
  {
    ROS_ERROR_STREAM("[camera] CameraInfo has no valid intrinsics (width=" << ci.width << ", K[0]=" << ci.K[0] << ")");
    return false;
  }
  auto D = [&](size_t i) { return i < ci.D.size() ? ci.D[i] : 0.0; };
  cam = makeCamera(ci.distortion_model, ci.width, ci.height, scale, ci.K[0], ci.K[4], ci.K[2], ci.K[5],
                   D(0), D(1), D(2), D(3));
  ROS_INFO_STREAM("[camera] online intrinsics from topic: "
                  << (ci.distortion_model.empty() ? "(none)" : ci.distortion_model) << " " << ci.width << "x"
                  << ci.height << "  fx=" << ci.K[0] << " fy=" << ci.K[4] << " cx=" << ci.K[2] << " cy=" << ci.K[5]
                  << "  D=[" << D(0) << ", " << D(1) << ", " << D(2) << ", " << D(3) << "]");
  return true;
}
} // namespace

LIVMapper::LIVMapper(ros::NodeHandle &nh)
    : extT(0, 0, 0),
      extR(M3D::Identity())
{
  extrinT.assign(3, 0.0);
  extrinR.assign(9, 0.0);
  cameraextrinT.assign(3, 0.0);
  cameraextrinR.assign(9, 0.0);

  p_pre.reset(new Preprocess());
  p_imu.reset(new ImuProcess());

  readParameters(nh);
  VoxelMapConfig voxel_config;
  loadVoxelConfig(nh, voxel_config);

  visual_sub_map.reset(new PointCloudXYZI());
  feats_undistort.reset(new PointCloudXYZI());
  feats_down_body.reset(new PointCloudXYZI());
  feats_down_world.reset(new PointCloudXYZI());
  pcl_w_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_save.reset(new PointCloudXYZRGB());
  pcl_wait_save_intensity.reset(new PointCloudXYZI());
  voxelmap_manager.reset(new VoxelMapManager(voxel_config, voxel_map));
  vio_manager.reset(new VIOManager());
  root_dir = ROOT_DIR;
  initializeFiles();
  initializeComponents();
  path.header.stamp = ros::Time::now();
  path.header.frame_id = "camera_init";
}

LIVMapper::~LIVMapper() {}

void LIVMapper::readParameters(ros::NodeHandle &nh)
{
  nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
  nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
  nh.param<bool>("common/ros_driver_bug_fix", ros_driver_fix_en, false);
  nh.param<int>("common/img_en", img_en, 1);
  nh.param<int>("common/lidar_en", lidar_en, 1);
  nh.param<bool>("common/imu_only_mode", imu_only_mode, false);   // skip LIO/VIO update -> publish pure IMU-propagated state (real fast-livo IMU-only dead-reckoning; needs lidar_en=1 to drive the loop + IMU init)
  nh.param<bool>("debug/fusion_log", fusion_debug, false);        // debug-only: log per-frame IMU-prop vs post-LIO/VIO attitude to /tmp/fusion_debug.csv
  nh.param<bool>("debug/vio_flip_roll", vio_flip_roll, false);    // debug-only: negate the VIO roll state update
  nh.param<bool>("debug/vio_flip_pitch", vio_flip_pitch, false);  // debug-only: negate the VIO pitch state update
  // -1 preserves upstream always-on fusion.  A non-negative value enables a
  // sensor-health fallback: VIO corrects the state only when the immediately
  // preceding LIO update has at most this many effective point-plane features.
  nh.param<int>("vio/max_lio_features_for_fusion",
                vio_max_lio_features_for_fusion, -1);
  nh.param<string>("common/img_topic", img_topic, "/left_camera/image");
  nh.param<bool>("common/online_intrinsics_en", online_intrinsics_en, false);
  nh.param<string>("common/cam_info_topic", cam_info_topic, "");
  nh.param<double>("common/cam_info_timeout", cam_info_timeout, 5.0);
  nh.param<string>("common/cam_calib_file", cam_calib_file, "");

  // OptiTrack/mocap gt-init: the first pose on this topic latches odom->camera_init
  // (gt_odom_cbk). Always subscribed; self-activates when mocap is publishing.
  nh.param<string>("mocap/gt_pose_topic", gt_pose_topic, "/vrpn_client_node/pure/pose");

  nh.param<bool>("vio/normal_en", normal_en, true);
  nh.param<bool>("vio/inverse_composition_en", inverse_composition_en, false);
  nh.param<int>("vio/max_iterations", max_iterations, 5);
  nh.param<double>("vio/img_point_cov", IMG_POINT_COV, 100);
  nh.param<bool>("vio/raycast_en", raycast_en, false);
  nh.param<bool>("vio/exposure_estimate_en", exposure_estimate_en, true);
  nh.param<double>("vio/inv_expo_cov", inv_expo_cov, 0.2);
  nh.param<int>("vio/grid_size", grid_size, 5);
  nh.param<int>("vio/grid_n_height", grid_n_height, 17);
  nh.param<int>("vio/patch_pyrimid_level", patch_pyrimid_level, 3);
  nh.param<int>("vio/patch_size", patch_size, 8);
  nh.param<double>("vio/outlier_threshold", outlier_threshold, 1000);

  nh.param<double>("time_offset/exposure_time_init", exposure_time_init, 0.0);
  nh.param<double>("time_offset/img_time_offset", img_time_offset, 0.0);
  nh.param<double>("time_offset/imu_time_offset", imu_time_offset, 0.0);
  nh.param<double>("time_offset/lidar_time_offset", lidar_time_offset, 0.0);
  nh.param<bool>("uav/imu_rate_odom", imu_prop_enable, false);
  nh.param<bool>("uav/gravity_align_en", gravity_align_en, false);

  nh.param<string>("evo/seq_name", seq_name, "01");
  nh.param<bool>("evo/pose_output_en", pose_output_en, false);
  nh.param<double>("imu/gyr_cov", gyr_cov, 1.0);
  nh.param<double>("imu/acc_cov", acc_cov, 1.0);
  nh.param<double>("imu/b_gyr_cov", b_gyr_cov, 0.0001);
  nh.param<double>("imu/b_acc_cov", b_acc_cov, 0.0001);
  nh.param<int>("imu/imu_int_frame", imu_int_frame, 3);
  nh.param<bool>("imu/imu_en", imu_en, false);
  nh.param<bool>("imu/gravity_est_en", gravity_est_en, true);
  nh.param<bool>("imu/ba_bg_est_en", ba_bg_est_en, true);

  nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
  nh.param<double>("preprocess/filter_size_surf", filter_size_surf_min, 0.5);
  nh.param<bool>("preprocess/hilti_en", hilti_en, false);
  nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
  nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 6);
  nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 3);
  nh.param<bool>("preprocess/feature_extract_enabled", p_pre->feature_enabled, false);

  nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
  nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
  nh.param<bool>("pcd_save/colmap_output_en", colmap_output_en, false);
  nh.param<double>("pcd_save/filter_size_pcd", filter_size_pcd, 0.5);
  nh.param<vector<double>>("extrin_calib/extrinsic_T", extrinT, vector<double>());
  nh.param<vector<double>>("extrin_calib/extrinsic_R", extrinR, vector<double>());
  nh.param<vector<double>>("extrin_calib/Pcl", cameraextrinT, vector<double>());
  nh.param<vector<double>>("extrin_calib/Rcl", cameraextrinR, vector<double>());
  nh.param<bool>("body_calib/enable", body_calib_en, false);
  nh.param<double>("body_calib/gt_avg_sec", gt_avg_sec, 3.0);
  {
    vector<double> qcb, tcb;
    nh.param<vector<double>>("body_calib/q_cam2body_xyzw", qcb, vector<double>());
    nh.param<vector<double>>("body_calib/t_cam2body", tcb, vector<double>());
    if (body_calib_en && qcb.size() == 4 && tcb.size() == 3) {
      Eigen::Quaterniond q(qcb[3], qcb[0], qcb[1], qcb[2]); q.normalize();
      R_cam2body = q.toRotationMatrix();
      t_cam2body << tcb[0], tcb[1], tcb[2];
      ROS_INFO("[body_calib] enabled: t_cam2body=[%.4f %.4f %.4f] m", tcb[0], tcb[1], tcb[2]);
    } else if (body_calib_en) {
      ROS_WARN("[body_calib] enable=true but q/t params missing/wrong size -> DISABLED");
      body_calib_en = false;
    }
  }
  nh.param<double>("debug/plot_time", plot_time, -10);
  nh.param<int>("debug/frame_cnt", frame_cnt, 6);
  nh.param<bool>("debug/verbose", verbose, false);   // per-frame VIO/LIO console spam on/off (set in launch, no rebuild)

  nh.param<double>("publish/blind_rgb_points", blind_rgb_points, 0.01);
  nh.param<int>("publish/pub_scan_num", pub_scan_num, 1);
  nh.param<bool>("publish/pub_effect_point_en", pub_effect_point_en, false);
  nh.param<bool>("publish/dense_map_en", dense_map_en, false);

  p_pre->blind_sqr = p_pre->blind * p_pre->blind;
}

void LIVMapper::initializeComponents() 
{
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
  extT << VEC_FROM_ARRAY(extrinT);
  extR << MAT_FROM_ARRAY(extrinR);

  voxelmap_manager->extT_ << VEC_FROM_ARRAY(extrinT);
  voxelmap_manager->extR_ << MAT_FROM_ARRAY(extrinR);
  voxelmap_manager->verbose = verbose;
  p_pre->verbose = verbose;

  // Camera intrinsics: prefer a one-shot CameraInfo from the live topic so they always
  // match the resolution the camera actually started at. Only if that topic is absent do
  // we fall back to the static calib file (common/cam_calib_file) -- and the file is NOT
  // rosparam-loaded by the launch; we load it here, on demand, so when the online path
  // works no (potentially stale) cam_* params ever land on the param server.
  bool cam_loaded = false;
  if (img_en != 0 && online_intrinsics_en)
  {
    std::string ci_topic = cam_info_topic;
    if (ci_topic.empty()) // derive "<prefix>/camera_info" from img_topic
    {
      size_t slash = img_topic.find_last_of('/');
      ci_topic = (slash == std::string::npos) ? "/camera/color/camera_info"
                                              : img_topic.substr(0, slash) + "/camera_info";
    }
    ROS_INFO_STREAM("[camera] trying online intrinsics: waiting up to " << cam_info_timeout << "s for CameraInfo on " << ci_topic);
    auto ci = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(ci_topic, ros::Duration(cam_info_timeout));
    if (ci)
    {
      double scale = 1.0;
      ros::param::get("laserMapping/scale", scale);
      cam_loaded = buildCameraFromInfo(*ci, scale, vio_manager->cam);
    }
    else if (!cam_calib_file.empty())
    {
      // No live topic -> load the named calib yaml into this node's namespace NOW, then
      // let vikit build the model from those params (same path other configs use).
      ROS_WARN_STREAM("[camera] no CameraInfo on " << ci_topic << " within " << cam_info_timeout
                      << "s; loading static intrinsics from " << cam_calib_file);
      std::string cmd = "rosparam load '" + cam_calib_file + "' /laserMapping";
      if (system(cmd.c_str()) != 0) ROS_ERROR_STREAM("[camera] '" << cmd << "' failed");
    }
  }
  if (!cam_loaded && !vk::camera_loader::loadFromRosNs("laserMapping", vio_manager->cam))
    throw std::runtime_error("Camera model not correctly specified (no CameraInfo, no usable calib file).");

  vio_manager->verbose = verbose;
  vio_manager->flip_roll = vio_flip_roll;
  vio_manager->flip_pitch = vio_flip_pitch;
  vio_manager->grid_size = grid_size;
  vio_manager->patch_size = patch_size;
  vio_manager->outlier_threshold = outlier_threshold;
  vio_manager->setImuToLidarExtrinsic(extT, extR);
  vio_manager->setLidarToCameraExtrinsic(cameraextrinR, cameraextrinT);
  vio_manager->state = &_state;
  vio_manager->state_propagat = &state_propagat;
  vio_manager->max_iterations = max_iterations;
  vio_manager->img_point_cov = IMG_POINT_COV;
  vio_manager->normal_en = normal_en;
  vio_manager->inverse_composition_en = inverse_composition_en;
  vio_manager->raycast_en = raycast_en;
  vio_manager->grid_n_width = grid_n_width;
  vio_manager->grid_n_height = grid_n_height;
  vio_manager->patch_pyrimid_level = patch_pyrimid_level;
  vio_manager->exposure_estimate_en = exposure_estimate_en;
  vio_manager->colmap_output_en = colmap_output_en;
  vio_manager->initializeVIO();

  p_imu->set_extrinsic(extT, extR);
  p_imu->set_gyr_cov_scale(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov_scale(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_inv_expo_cov(inv_expo_cov);
  p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
  p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
  p_imu->set_imu_init_frame_num(imu_int_frame);

  if (!imu_en) p_imu->disable_imu();
  if (!gravity_est_en) p_imu->disable_gravity_est();
  if (!ba_bg_est_en) p_imu->disable_bias_est();
  if (!exposure_estimate_en) p_imu->disable_exposure_est();

  slam_mode_ = (img_en && lidar_en) ? LIVO : imu_en ? ONLY_LIO : ONLY_LO;
}

void LIVMapper::initializeFiles() 
{
  if (pcd_save_en && colmap_output_en)
  {
      const std::string folderPath = std::string(ROOT_DIR) + "/scripts/colmap_output.sh";
      
      std::string chmodCommand = "chmod +x " + folderPath;
      
      int chmodRet = system(chmodCommand.c_str());  
      if (chmodRet != 0) {
          std::cerr << "Failed to set execute permissions for the script." << std::endl;
          return;
      }

      int executionRet = system(folderPath.c_str());
      if (executionRet != 0) {
          std::cerr << "Failed to execute the script." << std::endl;
          return;
      }
  }
  if(colmap_output_en) fout_points.open(std::string(ROOT_DIR) + "Log/Colmap/sparse/0/points3D.txt", std::ios::out);
  if(pcd_save_interval > 0) fout_pcd_pos.open(std::string(ROOT_DIR) + "Log/PCD/scans_pos.json", std::ios::out);
  fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), std::ios::out);
  fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);
}

void LIVMapper::initializeSubscribersAndPublishers(ros::NodeHandle &nh)
{
  sub_pcl = p_pre->lidar_type == AVIA ? 
            nh.subscribe(lid_topic, 10, &LIVMapper::livox_pcl_cbk, this): 
            nh.subscribe(lid_topic, 10, &LIVMapper::standard_pcl_cbk, this);  // [jetson] was 200000: tiny queue drops stale clouds under CPU load instead of buffering -> instant recovery after a stall (no slow backlog grind). See mapping_d435i.launch launch-prefix.
  sub_imu = nh.subscribe(imu_topic, 2000, &LIVMapper::imu_cbk, this);  // [jetson] was 200000: keep ~10s of IMU (cheap to drain, avoids propagation gaps) but bounded.
  sub_img = nh.subscribe(img_topic, 10, &LIVMapper::img_cbk, this);  // [jetson] was 200000: small queue, drop stale frames under load.
  sub_reinit = nh.subscribe("/livo/reinit", 1, &LIVMapper::reinit_cbk, this);  // re-anchor trigger
  // Always subscribe (like sim `ml`): the first pose that arrives latches odom->camera_init
  // (gt_odom_cbk), used by the imu_prop odom path; body_calib also buffers GT here to anchor
  // /aft_mapped_to_optitrack. Self-selects at runtime — no enable flag.
  sub_gt_odom = nh.subscribe(gt_pose_topic, 1, &LIVMapper::gt_odom_cbk, this);
  ROS_INFO("[mocap] gt-init armed: first pose on %s sets odom->camera_init", gt_pose_topic.c_str());

  pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
  pubNormal = nh.advertise<visualization_msgs::MarkerArray>("visualization_marker", 100);
  pubSubVisualMap = nh.advertise<sensor_msgs::PointCloud2>("/cloud_visual_sub_map_before", 100);
  pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);
  pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
  pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
  // /aft_mapped_to_odom is GONE -> renamed /aft_mapped_to_optitrack (GT-anchored body in OptiTrack frame).
  // /aft_mapped_to_body = the W_L body pose (PoseStamped) the estimator mux selects as the vio source.
  pubOdomAftMappedOdom = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_optitrack", 10);
  pubOdomAftMappedBody = nh.advertise<geometry_msgs::PoseStamped>("/aft_mapped_to_body", 10);
  pubPath = nh.advertise<nav_msgs::Path>("/path", 10);
  plane_pub = nh.advertise<visualization_msgs::Marker>("/planner_normal", 1);
  voxel_pub = nh.advertise<visualization_msgs::MarkerArray>("/voxels", 1);
  pubLaserCloudDyn = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj", 100);
  pubLaserCloudDynRmed = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj_removed", 100);
  pubLaserCloudDynDbg = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj_dbg_hist", 100);
  // /rgb_img is a BGR debug image, not a depth image.  Publishing it through
  // image_transport advertises every installed transport, including
  // /rgb_img/compressedDepth.  A record-all rosbag then subscribes to that
  // invalid transport and compressed_depth_image_transport logs an error for
  // every frame.  A plain Image publisher keeps the useful raw RViz topic and
  // does not advertise inapplicable depth transports.
  pubImage = nh.advertise<sensor_msgs::Image>("/rgb_img", 1);
  pubImuPropOdom = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_body_imu_propagated", 10000);
  imu_prop_timer = nh.createTimer(ros::Duration(0.004), &LIVMapper::imu_prop_callback, this);
  voxelmap_manager->voxel_map_pub_= nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);
}

void LIVMapper::handleFirstFrame() 
{
  if (!is_first_frame)
  {
    _first_lidar_time = LidarMeasures.last_lio_update_time;
    p_imu->first_lidar_time = _first_lidar_time; // Only for IMU data log
    is_first_frame = true;
    cout << "FIRST LIDAR FRAME!" << endl;
  }
}

void LIVMapper::gravityAlignment() 
{
  if (!p_imu->imu_need_init && !gravity_align_finished) 
  {
    std::cout << "Gravity Alignment Starts" << std::endl;
    V3D ez(0, 0, -1), gz(_state.gravity);
    Quaterniond G_q_I0 = Quaterniond::FromTwoVectors(gz, ez);
    M3D G_R_I0 = G_q_I0.toRotationMatrix();

    _state.pos_end = G_R_I0 * _state.pos_end;
    _state.rot_end = G_R_I0 * _state.rot_end;
    _state.vel_end = G_R_I0 * _state.vel_end;
    _state.gravity = G_R_I0 * _state.gravity;
    gravity_align_finished = true;
    std::cout << "Gravity Alignment Finished" << std::endl;
  }
}

void LIVMapper::processImu() 
{
  // double t0 = omp_get_wtime();

  p_imu->Process2(LidarMeasures, _state, feats_undistort);

  if (gravity_align_en) gravityAlignment();

  state_propagat = _state;
  voxelmap_manager->state_ = _state;
  voxelmap_manager->feats_undistort_ = feats_undistort;

  // double t_prop = omp_get_wtime();

  // std::cout << "[ Mapping ] feats_undistort: " << feats_undistort->size() << std::endl;
  // std::cout << "[ Mapping ] predict cov: " << _state.cov.diagonal().transpose() << std::endl;
  // std::cout << "[ Mapping ] predict sta: " << state_propagat.pos_end.transpose() << state_propagat.vel_end.transpose() << std::endl;
}

void LIVMapper::stateEstimationAndMapping() 
{
  switch (LidarMeasures.lio_vio_flg) 
  {
    case VIO:
      handleVIO();
      break;
    case LIO:
    case LO:
      handleLIO();
      break;
  }
}

void LIVMapper::handleVIO() 
{
  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << std::endl;
    
  if (pcl_w_wait_pub->empty() || (pcl_w_wait_pub == nullptr)) 
  {
    std::cout << "[ VIO ] No point!!!" << std::endl;
    return;
  }
    
  if (verbose) std::cout << "[ VIO ] Raw feature num: " << pcl_w_wait_pub->points.size() << std::endl;

  if (fabs((LidarMeasures.last_lio_update_time - _first_lidar_time) - plot_time) < (frame_cnt / 2 * 0.1)) 
  {
    vio_manager->plot_flag = true;
  } 
  else 
  {
    vio_manager->plot_flag = false;
  }

  vio_manager->state_update_enabled =
      vio_max_lio_features_for_fusion < 0 ||
      voxelmap_manager->effct_feat_num_ <= vio_max_lio_features_for_fusion;
  vio_manager->processFrame(LidarMeasures.measures.back().img, _pv_list, voxelmap_manager->voxel_map_, LidarMeasures.last_lio_update_time - _first_lidar_time);

  if (fusion_debug) {  // debug-only: log post-VIO internal quaternion (mapped to world offline) vs GT
    if (!dbg_fp) { dbg_fp = fopen("/tmp/fusion_debug.csv", "w"); fprintf(dbg_fp, "t,stage,pqx,pqy,pqz,pqw,qx,qy,qz,qw,nfeat,px,py,pz,rqx,rqy,rqz,rqw,vio_inlier_ratio,vio_error_ratio,lio_trans_info_ratio,lio_rot_info_ratio,lio_info_min_per_feature,vio_trans_info_ratio,vio_rot_info_ratio,vio_info_min_per_measurement\n"); }
    Eigen::Quaterniond qp(state_propagat.rot_end), qo(_state.rot_end), qr(vio_manager->raw_rot_vio_);
    fprintf(dbg_fp, "%.4f,VIO,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
            LidarMeasures.last_lio_update_time - _first_lidar_time, qp.x(), qp.y(), qp.z(), qp.w(), qo.x(), qo.y(), qo.z(), qo.w(),
            vio_manager->total_points, _state.pos_end[0], _state.pos_end[1], _state.pos_end[2], qr.x(), qr.y(), qr.z(), qr.w(),
            vio_manager->last_inlier_ratio, vio_manager->last_error_ratio,
            voxelmap_manager->last_translation_info_ratio_,
            voxelmap_manager->last_rotation_info_ratio_,
            voxelmap_manager->last_info_min_per_feature_,
            vio_manager->last_translation_info_ratio,
            vio_manager->last_rotation_info_ratio,
            vio_manager->last_info_min_per_measurement);
    fflush(dbg_fp);
  }

  if (imu_prop_enable) 
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  // monitor: fill /cloud_visual_sub_map_before from the ACTIVE visual_submap (upstream fill was disabled)
  visual_sub_map->clear();
  if (vio_manager->visual_submap != nullptr)
    for (VisualPoint *vp : vio_manager->visual_submap->voxel_points)
    {
      if (vp == nullptr) continue;
      PointType temp_map;
      temp_map.x = vp->pos_[0]; temp_map.y = vp->pos_[1]; temp_map.z = vp->pos_[2]; temp_map.intensity = 0.;
      visual_sub_map->push_back(temp_map);
    }
  publish_visual_sub_map(pubSubVisualMap);

  publish_frame_world(pubLaserCloudFullRes, vio_manager);
  publish_img_rgb(pubImage, vio_manager);

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;
}

void LIVMapper::handleLIO() 
{    
  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
           << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
           << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << endl;
           
  if (imu_only_mode)  // real IMU-only: _state here is already the IMU-propagated state (Process2); publish it, skip LIO update + map
  {
    if (imu_prop_enable) { ekf_finish_once = true; latest_ekf_state = _state; latest_ekf_time = LidarMeasures.last_lio_update_time; state_update_flg = true; }
    euler_cur = RotMtoEuler(_state.rot_end);
    geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
    publish_odometry(pubOdomAftMapped);
    publish_body_optitrack();
    return;
  }

  if (feats_undistort->empty() || (feats_undistort == nullptr))
  {
    std::cout << "[ LIO ]: No point!!!" << std::endl;
    return;
  }

  double t0 = omp_get_wtime();

  downSizeFilterSurf.setInputCloud(feats_undistort);
  downSizeFilterSurf.filter(*feats_down_body);
  
  double t_down = omp_get_wtime();

  feats_down_size = feats_down_body->points.size();
  voxelmap_manager->feats_down_body_ = feats_down_body;
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, feats_down_world);
  voxelmap_manager->feats_down_world_ = feats_down_world;
  voxelmap_manager->feats_down_size_ = feats_down_size;
  
  if (!lidar_map_inited) 
  {
    lidar_map_inited = true;
    voxelmap_manager->BuildVoxelMap();
  }

  double t1 = omp_get_wtime();

  voxelmap_manager->StateEstimation(state_propagat);
  _state = voxelmap_manager->state_;
  _pv_list = voxelmap_manager->pv_list_;

  if (fusion_debug) {  // debug-only: log post-LIO internal quaternion (mapped to world offline) vs GT
    if (!dbg_fp) { dbg_fp = fopen("/tmp/fusion_debug.csv", "w"); fprintf(dbg_fp, "t,stage,pqx,pqy,pqz,pqw,qx,qy,qz,qw,nfeat,px,py,pz,rqx,rqy,rqz,rqw,vio_inlier_ratio,vio_error_ratio,lio_trans_info_ratio,lio_rot_info_ratio,lio_info_min_per_feature,vio_trans_info_ratio,vio_rot_info_ratio,vio_info_min_per_measurement\n"); }
    Eigen::Quaterniond qp(state_propagat.rot_end), qo(_state.rot_end), qr(voxelmap_manager->raw_rot_lio_);
    fprintf(dbg_fp, "%.4f,LIO,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
            LidarMeasures.last_lio_update_time - _first_lidar_time, qp.x(), qp.y(), qp.z(), qp.w(), qo.x(), qo.y(), qo.z(), qo.w(),
            voxelmap_manager->effct_feat_num_, _state.pos_end[0], _state.pos_end[1], _state.pos_end[2], qr.x(), qr.y(), qr.z(), qr.w(),
            vio_manager->last_inlier_ratio, vio_manager->last_error_ratio,
            voxelmap_manager->last_translation_info_ratio_,
            voxelmap_manager->last_rotation_info_ratio_,
            voxelmap_manager->last_info_min_per_feature_,
            vio_manager->last_translation_info_ratio,
            vio_manager->last_rotation_info_ratio,
            vio_manager->last_info_min_per_measurement);
    fflush(dbg_fp);
  }

  double t2 = omp_get_wtime();

  if (imu_prop_enable) 
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  if (pose_output_en) 
  {
    static bool pos_opend = false;
    static int ocount = 0;
    std::ofstream outFile, evoFile;
    if (!pos_opend) 
    {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt", std::ios::out);
      pos_opend = true;
      if (!evoFile.is_open()) ROS_ERROR("open fail\n");
    } 
    else 
    {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt", std::ios::app);
      if (!evoFile.is_open()) ROS_ERROR("open fail\n");
    }
    Eigen::Matrix4d outT;
    Eigen::Quaterniond q(_state.rot_end);
    evoFile << std::fixed;
    evoFile << LidarMeasures.last_lio_update_time << " " << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
  
  euler_cur = RotMtoEuler(_state.rot_end);
  geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
  publish_odometry(pubOdomAftMapped);
  publish_body_optitrack();

  double t3 = omp_get_wtime();

  PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI());
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, world_lidar);
  for (size_t i = 0; i < world_lidar->points.size(); i++) 
  {
    voxelmap_manager->pv_list_[i].point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
    M3D point_crossmat = voxelmap_manager->cross_mat_list_[i];
    M3D var = voxelmap_manager->body_cov_list_[i];
    var = (_state.rot_end * extR) * var * (_state.rot_end * extR).transpose() +
          (-point_crossmat) * _state.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + _state.cov.block<3, 3>(3, 3);
    voxelmap_manager->pv_list_[i].var = var;
  }
  voxelmap_manager->UpdateVoxelMap(voxelmap_manager->pv_list_);
  if (verbose) std::cout << "[ LIO ] Update Voxel Map" << std::endl;
  _pv_list = voxelmap_manager->pv_list_;
  
  double t4 = omp_get_wtime();

  if(voxelmap_manager->config_setting_.map_sliding_en)
  {
    voxelmap_manager->mapSliding();
  }
  
  PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort : feats_down_body);
  int size = laserCloudFullRes->points.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++) 
  {
    RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
  }
  *pcl_w_wait_pub = *laserCloudWorld;

  if (!img_en) publish_frame_world(pubLaserCloudFullRes, vio_manager);
  if (pub_effect_point_en) publish_effect_world(pubLaserCloudEffect, voxelmap_manager->ptpl_list_);
  if (voxelmap_manager->config_setting_.is_pub_plane_map_) voxelmap_manager->pubVoxelMap();
  publish_path(pubPath);
  // The estimator mux (vision_pose_mux.launch) is the SOLE publisher of /mavros/vision_pose/pose; it
  // selects /aft_mapped_to_body (vio) or the VRPN pose (mocap). fast-livo never writes that topic itself.

  frame_num++;
  aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t4 - t0) / frame_num;

  // aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t2 - t1) / frame_num;
  // aver_time_map_inre = aver_time_map_inre * (frame_num - 1) / frame_num + (t4 - t3) / frame_num;
  // aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time) / frame_num;
  // aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_const_H_time / frame_num;
  // printf("[ mapping time ]: per scan: propagation %0.6f downsample: %0.6f match: %0.6f solve: %0.6f  ICP: %0.6f  map incre: %0.6f total: %0.6f \n"
  //         "[ mapping time ]: average: icp: %0.6f construct H: %0.6f, total: %0.6f \n",
  //         t_prop - t0, t1 - t_prop, match_time, solve_time, t3 - t1, t5 - t3, t5 - t0, aver_time_icp, aver_time_const_H_time, aver_time_consu);

  // printf("\033[1;36m[ LIO mapping time ]: current scan: icp: %0.6f secs, map incre: %0.6f secs, total: %0.6f secs.\033[0m\n"
  //         "\033[1;36m[ LIO mapping time ]: average: icp: %0.6f secs, map incre: %0.6f secs, total: %0.6f secs.\033[0m\n",
  //         t2 - t1, t4 - t3, t4 - t0, aver_time_icp, aver_time_map_inre, aver_time_consu);
  if (verbose)
  {
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         LIO Mapping Time                    |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "DownSample", t_down - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "ICP", t2 - t1);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "updateVoxelMap", t4 - t3);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Current Total Time", t4 - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Average Total Time", aver_time_consu);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  }

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;
}

void LIVMapper::savePCD() 
{
  if (pcd_save_en && (pcl_wait_save->points.size() > 0 || pcl_wait_save_intensity->points.size() > 0) && pcd_save_interval < 0) 
  {
    std::string raw_points_dir = std::string(ROOT_DIR) + "Log/PCD/all_raw_points.pcd";
    std::string downsampled_points_dir = std::string(ROOT_DIR) + "Log/PCD/all_downsampled_points.pcd";
    pcl::PCDWriter pcd_writer;

    if (img_en)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
      voxel_filter.setInputCloud(pcl_wait_save);
      voxel_filter.setLeafSize(filter_size_pcd, filter_size_pcd, filter_size_pcd);
      voxel_filter.filter(*downsampled_cloud);
  
      pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save); // Save the raw point cloud data
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir 
                << " with point count: " << pcl_wait_save->points.size() << RESET << std::endl;
      
      pcd_writer.writeBinary(downsampled_points_dir, *downsampled_cloud); // Save the downsampled point cloud data
      std::cout << GREEN << "Downsampled point cloud data saved to: " << downsampled_points_dir 
                << " with point count after filtering: " << downsampled_cloud->points.size() << RESET << std::endl;

      if(colmap_output_en)
      {
        fout_points << "# 3D point list with one line of data per point\n";
        fout_points << "#  POINT_ID, X, Y, Z, R, G, B, ERROR\n";
        for (size_t i = 0; i < downsampled_cloud->size(); ++i) 
        {
            const auto& point = downsampled_cloud->points[i];
            fout_points << i << " "
                        << std::fixed << std::setprecision(6)
                        << point.x << " " << point.y << " " << point.z << " "
                        << static_cast<int>(point.r) << " "
                        << static_cast<int>(point.g) << " "
                        << static_cast<int>(point.b) << " "
                        << 0 << std::endl;
        }
      }
    }
    else
    {      
      pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save_intensity);
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir 
                << " with point count: " << pcl_wait_save_intensity->points.size() << RESET << std::endl;
    }
  }
}

void LIVMapper::run() 
{
  ros::Rate rate(5000);
  while (ros::ok()) 
  {
    ros::spinOnce();
    if (!sync_packages(LidarMeasures)) 
    {
      rate.sleep();
      continue;
    }
    handleFirstFrame();

    processImu();

    // if (!p_imu->imu_time_init) continue;

    stateEstimationAndMapping();
  }
  savePCD();
}

void LIVMapper::prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr)
{
  double mean_acc_norm = p_imu->IMU_mean_acc_norm;
  acc_avr = acc_avr * G_m_s2 / mean_acc_norm - imu_prop_state.bias_a;
  angvel_avr -= imu_prop_state.bias_g;

  M3D Exp_f = Exp(angvel_avr, dt);
  /* propogation of IMU attitude */
  imu_prop_state.rot_end = imu_prop_state.rot_end * Exp_f;

  /* Specific acceleration (global frame) of IMU */
  V3D acc_imu = imu_prop_state.rot_end * acc_avr + V3D(imu_prop_state.gravity[0], imu_prop_state.gravity[1], imu_prop_state.gravity[2]);

  /* propogation of IMU */
  imu_prop_state.pos_end = imu_prop_state.pos_end + imu_prop_state.vel_end * dt + 0.5 * acc_imu * dt * dt;

  /* velocity of IMU */
  imu_prop_state.vel_end = imu_prop_state.vel_end + acc_imu * dt;
}

void LIVMapper::imu_prop_callback(const ros::TimerEvent &e)
{
  if (p_imu->imu_need_init || !new_imu || !ekf_finish_once) { return; }
  mtx_buffer_imu_prop.lock();
  new_imu = false; // 控制propagate频率和IMU频率一致
  if (imu_prop_enable && !prop_imu_buffer.empty())
  {
    static double last_t_from_lidar_end_time = 0;
    if (state_update_flg)
    {
      imu_propagate = latest_ekf_state;
      // drop all useless imu pkg
      while ((!prop_imu_buffer.empty() && prop_imu_buffer.front().header.stamp.toSec() < latest_ekf_time))
      {
        prop_imu_buffer.pop_front();
      }
      last_t_from_lidar_end_time = 0;
      for (int i = 0; i < prop_imu_buffer.size(); i++)
      {
        double t_from_lidar_end_time = prop_imu_buffer[i].header.stamp.toSec() - latest_ekf_time;
        double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
        // cout << "prop dt" << dt << ", " << t_from_lidar_end_time << ", " << last_t_from_lidar_end_time << endl;
        V3D acc_imu(prop_imu_buffer[i].linear_acceleration.x, prop_imu_buffer[i].linear_acceleration.y, prop_imu_buffer[i].linear_acceleration.z);
        V3D omg_imu(prop_imu_buffer[i].angular_velocity.x, prop_imu_buffer[i].angular_velocity.y, prop_imu_buffer[i].angular_velocity.z);
        prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
        last_t_from_lidar_end_time = t_from_lidar_end_time;
      }
      state_update_flg = false;
    }
    else
    {
      V3D acc_imu(newest_imu.linear_acceleration.x, newest_imu.linear_acceleration.y, newest_imu.linear_acceleration.z);
      V3D omg_imu(newest_imu.angular_velocity.x, newest_imu.angular_velocity.y, newest_imu.angular_velocity.z);
      double t_from_lidar_end_time = newest_imu.header.stamp.toSec() - latest_ekf_time;
      double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
      prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
      last_t_from_lidar_end_time = t_from_lidar_end_time;
    }

    V3D posi, vel_i;
    Eigen::Quaterniond q;
    posi = imu_propagate.pos_end;
    vel_i = imu_propagate.vel_end;
    q = Eigen::Quaterniond(imu_propagate.rot_end);
    imu_prop_odom.header.stamp = newest_imu.header.stamp;
    imu_prop_odom.child_frame_id = "base_link";
    if (body_calib_en && vio_manager) {
      // Publish the BODY(marker) pose, not the raw IMU, so /aft_mapped_to_body_imu_propagated matches
      // /aft_mapped_to_body. The planner reads YAW from this odom (state_manager.py); the
      // raw IMU optical frame gives a ~90deg-wrong heading and disagrees with the EKF2/
      // control body frame. Same hand-eye chain as publish_body_optitrack, plus twist, at
      // IMU rate. World bridge: optitrack anchor when latched (global == /aft_mapped_to_optitrack),
      // else the q_o2r optical->ROS flip (VIO-local == /aft_mapped_to_body).
      Eigen::Isometry3d T_WI = Eigen::Isometry3d::Identity();
      T_WI.linear() = imu_propagate.rot_end;  T_WI.translation() = imu_propagate.pos_end;
      Eigen::Isometry3d T_CI = Eigen::Isometry3d::Identity();
      T_CI.linear() = vio_manager->Rci;  T_CI.translation() = vio_manager->Pci;
      Eigen::Isometry3d T_BC = Eigen::Isometry3d::Identity();
      T_BC.linear() = R_cam2body;  T_BC.translation() = t_cam2body;
      Eigen::Isometry3d T_WB = T_WI * T_CI.inverse() * T_BC.inverse();

      Eigen::Isometry3d T_bridge = Eigen::Isometry3d::Identity();
      if (body_anchor_latched) T_bridge = T_anchor;                                       // global (optitrack)
      else T_bridge.linear() = Eigen::Quaterniond(-0.5, 0.5, -0.5, 0.5).toRotationMatrix();// q_o2r (w,x,y,z)
      Eigen::Isometry3d T_pub = T_bridge * T_WB;
      Eigen::Vector3d   v_pub = T_bridge.linear() * vel_i;   // world-frame vel; marker lever-arm ignored (small)
      Eigen::Quaterniond q_pub(T_pub.linear());

      imu_prop_odom.header.frame_id = "odom";
      imu_prop_odom.pose.pose.position.x = T_pub.translation().x();
      imu_prop_odom.pose.pose.position.y = T_pub.translation().y();
      imu_prop_odom.pose.pose.position.z = T_pub.translation().z();
      imu_prop_odom.pose.pose.orientation.x = q_pub.x();
      imu_prop_odom.pose.pose.orientation.y = q_pub.y();
      imu_prop_odom.pose.pose.orientation.z = q_pub.z();
      imu_prop_odom.pose.pose.orientation.w = q_pub.w();
      imu_prop_odom.twist.twist.linear.x = v_pub.x();
      imu_prop_odom.twist.twist.linear.y = v_pub.y();
      imu_prop_odom.twist.twist.linear.z = v_pub.z();
    } else if (gt_odom_received) {
      // Legacy IMU gt-latch (body_calib off): odom_pose = odom_to_camera_init * camera_init_pose,
      // so /aft_mapped_to_body_imu_propagated starts at the true mocap pose (matches the sim `ml` branch).
      tf::Transform odom_to_cam_init_tf;
      odom_to_cam_init_tf.setOrigin(tf::Vector3(odom_to_camera_init.translation.x,
                                                odom_to_camera_init.translation.y,
                                                odom_to_camera_init.translation.z));
      tf::Quaternion q_otc(odom_to_camera_init.rotation.x, odom_to_camera_init.rotation.y,
                           odom_to_camera_init.rotation.z, odom_to_camera_init.rotation.w);
      odom_to_cam_init_tf.setRotation(q_otc);
      tf::Transform cam_init_pose_tf;
      cam_init_pose_tf.setOrigin(tf::Vector3(posi.x(), posi.y(), posi.z()));
      cam_init_pose_tf.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));
      tf::Transform odom_pose_tf = odom_to_cam_init_tf * cam_init_pose_tf;
      tf::Vector3 p = odom_pose_tf.getOrigin();
      tf::Quaternion qt = odom_pose_tf.getRotation();
      imu_prop_odom.header.frame_id = "odom";
      imu_prop_odom.pose.pose.position.x = p.x();
      imu_prop_odom.pose.pose.position.y = p.y();
      imu_prop_odom.pose.pose.position.z = p.z();
      imu_prop_odom.pose.pose.orientation.x = qt.x();
      imu_prop_odom.pose.pose.orientation.y = qt.y();
      imu_prop_odom.pose.pose.orientation.z = qt.z();
      imu_prop_odom.pose.pose.orientation.w = qt.w();
      tf::Vector3 vel_odom = tf::quatRotate(q_otc, tf::Vector3(vel_i.x(), vel_i.y(), vel_i.z()));
      imu_prop_odom.twist.twist.linear.x = vel_odom.x();
      imu_prop_odom.twist.twist.linear.y = vel_odom.y();
      imu_prop_odom.twist.twist.linear.z = vel_odom.z();
    } else {
      imu_prop_odom.header.frame_id = "world";
      imu_prop_odom.pose.pose.position.x = posi.x();
      imu_prop_odom.pose.pose.position.y = posi.y();
      imu_prop_odom.pose.pose.position.z = posi.z();
      imu_prop_odom.pose.pose.orientation.w = q.w();
      imu_prop_odom.pose.pose.orientation.x = q.x();
      imu_prop_odom.pose.pose.orientation.y = q.y();
      imu_prop_odom.pose.pose.orientation.z = q.z();
      imu_prop_odom.twist.twist.linear.x = vel_i.x();
      imu_prop_odom.twist.twist.linear.y = vel_i.y();
      imu_prop_odom.twist.twist.linear.z = vel_i.z();
    }
    pubImuPropOdom.publish(imu_prop_odom);
  }
  mtx_buffer_imu_prop.unlock();
}

void LIVMapper::transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud)
{
  PointCloudXYZI().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR * p + extT) + t);
    PointType pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}

void LIVMapper::pointBodyToWorld(const PointType &pi, PointType &po)
{
  V3D p_body(pi.x, pi.y, pi.z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po.x = p_global(0);
  po.y = p_global(1);
  po.z = p_global(2);
  po.intensity = pi.intensity;
}

template <typename T> void LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

template <typename T> Matrix<T, 3, 1> LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi)
{
  V3D p(pi[0], pi[1], pi[2]);
  p = (_state.rot_end * (extR * p + extT) + _state.pos_end);
  Matrix<T, 3, 1> po(p[0], p[1], p[2]);
  return po;
}

void LIVMapper::RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LIVMapper::standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  if (!lidar_en) return;
  mtx_buffer.lock();

  double cur_head_time = msg->header.stamp.toSec() + lidar_time_offset;
  // cout<<"got feature"<<endl;
  if (cur_head_time < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LIVMapper::livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in)
{
  if (!lidar_en) return;
  mtx_buffer.lock();
  livox_ros_driver::CustomMsg::Ptr msg(new livox_ros_driver::CustomMsg(*msg_in));
  // if ((abs(msg->header.stamp.toSec() - last_timestamp_lidar) > 0.2 && last_timestamp_lidar > 0) || sync_jump_flag)
  // {
  //   ROS_WARN("lidar jumps %.3f\n", msg->header.stamp.toSec() - last_timestamp_lidar);
  //   sync_jump_flag = true;
  //   msg->header.stamp = ros::Time().fromSec(last_timestamp_lidar + 0.1);
  // }
  if (abs(last_timestamp_imu - msg->header.stamp.toSec()) > 1.0 && !imu_buffer.empty())
  {
    double timediff_imu_wrt_lidar = last_timestamp_imu - msg->header.stamp.toSec();
    printf("\033[95mSelf sync IMU and LiDAR, HARD time lag is %.10lf \n\033[0m", timediff_imu_wrt_lidar - 0.100);
    // imu_time_offset = timediff_imu_wrt_lidar;
  }

  double cur_head_time = msg->header.stamp.toSec();
  ROS_INFO("Get LiDAR, its header time: %.6f", cur_head_time);
  if (cur_head_time < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);

  if (!ptr || ptr->empty()) {
    ROS_ERROR("Received an empty point cloud");
    mtx_buffer.unlock();
    return;
  }

  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LIVMapper::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
  if (!imu_en) return;

  if (last_timestamp_lidar < 0.0) return;
  // ROS_INFO("get imu at time: %.6f", msg_in->header.stamp.toSec());
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
  msg->header.stamp = ros::Time().fromSec(msg->header.stamp.toSec() - imu_time_offset);
  double timestamp = msg->header.stamp.toSec();

  if (fabs(last_timestamp_lidar - timestamp) > 0.5 && (!ros_driver_fix_en))
  {
    ROS_WARN("IMU and LiDAR not synced! delta time: %lf .\n", last_timestamp_lidar - timestamp);
  }

  if (ros_driver_fix_en) timestamp += std::round(last_timestamp_lidar - timestamp);
  msg->header.stamp = ros::Time().fromSec(timestamp);

  mtx_buffer.lock();

  if (last_timestamp_imu > 0.0 && timestamp < last_timestamp_imu)
  {
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    ROS_ERROR("imu loop back, offset: %lf \n", last_timestamp_imu - timestamp);
    return;
  }

  // if (last_timestamp_imu > 0.0 && timestamp > last_timestamp_imu + 0.2)
  // {

  //   ROS_WARN("imu time stamp Jumps %0.4lf seconds \n", timestamp - last_timestamp_imu);
  //   mtx_buffer.unlock();
  //   sig_buffer.notify_all();
  //   return;
  // }

  last_timestamp_imu = timestamp;

  imu_buffer.push_back(msg);
  // cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<endl;
  mtx_buffer.unlock();
  if (imu_prop_enable)
  {
    mtx_buffer_imu_prop.lock();
    if (imu_prop_enable && !p_imu->imu_need_init) { prop_imu_buffer.push_back(*msg); }
    newest_imu = *msg;
    new_imu = true;
    mtx_buffer_imu_prop.unlock();
  }
  sig_buffer.notify_all();
}

cv::Mat LIVMapper::getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
  cv::Mat img;
  img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
  return img;
}

void LIVMapper::img_cbk(const sensor_msgs::ImageConstPtr &msg_in)
{
  if (!img_en) return;
  sensor_msgs::Image::Ptr msg(new sensor_msgs::Image(*msg_in));
  // if ((abs(msg->header.stamp.toSec() - last_timestamp_img) > 0.2 && last_timestamp_img > 0) || sync_jump_flag)
  // {
  //   ROS_WARN("img jumps %.3f\n", msg->header.stamp.toSec() - last_timestamp_img);
  //   sync_jump_flag = true;
  //   msg->header.stamp = ros::Time().fromSec(last_timestamp_img + 0.1);
  // }

  // Hiliti2022 40Hz
  if (hilti_en)
  {
    static int frame_counter = 0;
    if (++frame_counter % 4 != 0) return;
  }
  // double msg_header_time =  msg->header.stamp.toSec();
  double msg_header_time = msg->header.stamp.toSec() + img_time_offset;
  if (abs(msg_header_time - last_timestamp_img) < 0.001) return;
  if (verbose) ROS_INFO("Get image, its header time: %.6f", msg_header_time);
  if (last_timestamp_lidar < 0) return;

  if (msg_header_time < last_timestamp_img)
  {
    ROS_ERROR("image loop back. \n");
    return;
  }

  mtx_buffer.lock();

  double img_time_correct = msg_header_time; // last_timestamp_lidar + 0.105;

  if (img_time_correct - last_timestamp_img < 0.02)
  {
    ROS_WARN("Image need Jumps: %.6f", img_time_correct);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    return;
  }

  cv::Mat img_cur = getImageFromMsg(msg);
  img_buffer.push_back(img_cur);
  img_time_buffer.push_back(img_time_correct);

  // ROS_INFO("Correct Image time: %.6f", img_time_correct);

  last_timestamp_img = img_time_correct;
  // cv::imshow("img", img);
  // cv::waitKey(1);
  // cout<<"last_timestamp_img:::"<<last_timestamp_img<<endl;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

bool LIVMapper::sync_packages(LidarMeasureGroup &meas)
{
  if (lid_raw_data_buffer.empty() && lidar_en) return false;
  if (img_buffer.empty() && img_en) return false;
  if (imu_buffer.empty() && imu_en) return false;

  switch (slam_mode_)
  {
  case ONLY_LIO:
  {
    if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();
    if (!lidar_pushed)
    {
      // If not push the lidar into measurement data buffer
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      if (meas.lidar->points.size() <= 1) return false;

      meas.lidar_frame_beg_time = lid_header_time_buffer.front();                                                // generate lidar_frame_beg_time
      meas.lidar_frame_end_time = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      meas.pcl_proc_cur = meas.lidar;
      lidar_pushed = true;                                                                                       // flag
    }

    if (imu_en && last_timestamp_imu < meas.lidar_frame_end_time)
    { // waiting imu message needs to be
      // larger than _lidar_frame_end_time,
      // make sure complete propagate.
      // ROS_ERROR("out sync");
      return false;
    }

    struct MeasureGroup m; // standard method to keep imu message.

    m.imu.clear();
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    while (!imu_buffer.empty())
    {
      if (imu_buffer.front()->header.stamp.toSec() > meas.lidar_frame_end_time) break;
      m.imu.push_back(imu_buffer.front());
      imu_buffer.pop_front();
    }
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();

    meas.lio_vio_flg = LIO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    // ROS_INFO("ONlY HAS LiDAR and IMU, NO IMAGE!");
    lidar_pushed = false; // sync one whole lidar scan.
    return true;

    break;
  }

  case LIVO:
  {
    /*** For LIVO mode, the time of LIO update is set to be the same as VIO, LIO
     * first than VIO imediatly ***/
    EKF_STATE last_lio_vio_flg = meas.lio_vio_flg;
    // double t0 = omp_get_wtime();
    switch (last_lio_vio_flg)
    {
    // double img_capture_time = meas.lidar_frame_beg_time + exposure_time_init;
    case WAIT:
    case VIO:
    {
      // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
      double img_capture_time = img_time_buffer.front() + exposure_time_init;
      /*** has img topic, but img topic timestamp larger than lidar end time,
       * process lidar topic. After LIO update, the meas.lidar_frame_end_time
       * will be refresh. ***/
      if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();
      // printf("[ Data Cut ] wait \n");
      // printf("[ Data Cut ] last_lio_update_time: %lf \n",
      // meas.last_lio_update_time);

      double lid_newest_time = lid_header_time_buffer.back() + lid_raw_data_buffer.back()->points.back().curvature / double(1000);
      double imu_newest_time = imu_buffer.back()->header.stamp.toSec();

      if (img_capture_time < meas.last_lio_update_time + 0.00001)
      {
        img_buffer.pop_front();
        img_time_buffer.pop_front();
        ROS_ERROR("[ Data Cut ] Throw one image frame! \n");
        return false;
      }

      if (img_capture_time > lid_newest_time || img_capture_time > imu_newest_time)
      {
        // ROS_ERROR("lost first camera frame");
        // printf("img_capture_time, lid_newest_time, imu_newest_time: %lf , %lf
        // , %lf \n", img_capture_time, lid_newest_time, imu_newest_time);
        return false;
      }

      struct MeasureGroup m;

      // printf("[ Data Cut ] LIO \n");
      // printf("[ Data Cut ] img_capture_time: %lf \n", img_capture_time);
      m.imu.clear();
      m.lio_time = img_capture_time;
      mtx_buffer.lock();
      while (!imu_buffer.empty())
      {
        if (imu_buffer.front()->header.stamp.toSec() > m.lio_time) break;

        if (imu_buffer.front()->header.stamp.toSec() > meas.last_lio_update_time) m.imu.push_back(imu_buffer.front());

        imu_buffer.pop_front();
        // printf("[ Data Cut ] imu time: %lf \n",
        // imu_buffer.front()->header.stamp.toSec());
      }
      mtx_buffer.unlock();
      sig_buffer.notify_all();

      *(meas.pcl_proc_cur) = *(meas.pcl_proc_next);
      PointCloudXYZI().swap(*meas.pcl_proc_next);

      int lid_frame_num = lid_raw_data_buffer.size();
      int max_size = meas.pcl_proc_cur->size() + 24000 * lid_frame_num;
      meas.pcl_proc_cur->reserve(max_size);
      meas.pcl_proc_next->reserve(max_size);
      // deque<PointCloudXYZI::Ptr> lidar_buffer_tmp;

      while (!lid_raw_data_buffer.empty())
      {
        if (lid_header_time_buffer.front() > img_capture_time) break;
        auto pcl(lid_raw_data_buffer.front()->points);
        double frame_header_time(lid_header_time_buffer.front());
        float max_offs_time_ms = (m.lio_time - frame_header_time) * 1000.0f;

        for (int i = 0; i < pcl.size(); i++)
        {
          auto pt = pcl[i];
          if (pcl[i].curvature < max_offs_time_ms)
          {
            pt.curvature += (frame_header_time - meas.last_lio_update_time) * 1000.0f;
            meas.pcl_proc_cur->points.push_back(pt);
          }
          else
          {
            pt.curvature += (frame_header_time - m.lio_time) * 1000.0f;
            meas.pcl_proc_next->points.push_back(pt);
          }
        }
        lid_raw_data_buffer.pop_front();
        lid_header_time_buffer.pop_front();
      }

      meas.measures.push_back(m);
      meas.lio_vio_flg = LIO;
      // meas.last_lio_update_time = m.lio_time;
      // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
      // printf("[ Data Cut ] pcl_proc_cur number: %d \n", meas.pcl_proc_cur
      // ->points.size()); printf("[ Data Cut ] LIO process time: %lf \n",
      // omp_get_wtime() - t0);
      return true;
    }

    case LIO:
    {
      double img_capture_time = img_time_buffer.front() + exposure_time_init;
      meas.lio_vio_flg = VIO;
      // printf("[ Data Cut ] VIO \n");
      meas.measures.clear();
      double imu_time = imu_buffer.front()->header.stamp.toSec();

      struct MeasureGroup m;
      m.vio_time = img_capture_time;
      m.lio_time = meas.last_lio_update_time;
      m.img = img_buffer.front();
      mtx_buffer.lock();
      // while ((!imu_buffer.empty() && (imu_time < img_capture_time)))
      // {
      //   imu_time = imu_buffer.front()->header.stamp.toSec();
      //   if (imu_time > img_capture_time) break;
      //   m.imu.push_back(imu_buffer.front());
      //   imu_buffer.pop_front();
      //   printf("[ Data Cut ] imu time: %lf \n",
      //   imu_buffer.front()->header.stamp.toSec());
      // }
      img_buffer.pop_front();
      img_time_buffer.pop_front();
      mtx_buffer.unlock();
      sig_buffer.notify_all();
      meas.measures.push_back(m);
      lidar_pushed = false; // after VIO update, the _lidar_frame_end_time will be refresh.
      // printf("[ Data Cut ] VIO process time: %lf \n", omp_get_wtime() - t0);
      return true;
    }

    default:
    {
      // printf("!! WRONG EKF STATE !!");
      return false;
    }
      // return false;
    }
    break;
  }

  case ONLY_LO:
  {
    if (!lidar_pushed) 
    { 
      // If not in lidar scan, need to generate new meas
      if (lid_raw_data_buffer.empty())  return false;
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      meas.lidar_frame_beg_time = lid_header_time_buffer.front(); // generate lidar_beg_time
      meas.lidar_frame_end_time  = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      lidar_pushed = true;             
    }
    struct MeasureGroup m; // standard method to keep imu message.
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    lidar_pushed = false; // sync one whole lidar scan.
    meas.lio_vio_flg = LO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    return true;
    break;
  }

  default:
  {
    printf("!! WRONG SLAM TYPE !!");
    return false;
  }
  }
  ROS_ERROR("out sync");
}

void LIVMapper::publish_img_rgb(const ros::Publisher &pubImage, VIOManagerPtr vio_manager)
{
  cv::Mat img_rgb = vio_manager->img_cp;
  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = ros::Time::now();
  // out_msg.header.frame_id = "camera_init";
  out_msg.encoding = sensor_msgs::image_encodings::BGR8;
  out_msg.image = img_rgb;
  pubImage.publish(out_msg.toImageMsg());
}

void LIVMapper::publish_frame_world(const ros::Publisher &pubLaserCloudFullRes, VIOManagerPtr vio_manager)
{
  if (pcl_w_wait_pub->empty()) return;
  PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB());
  if (img_en)
  {
    static int pub_num = 1;
    *pcl_wait_pub += *pcl_w_wait_pub;
    if(pub_num == pub_scan_num)
    {
      pub_num = 1;
      size_t size = pcl_wait_pub->points.size();
      laserCloudWorldRGB->reserve(size);
      // double inv_expo = _state.inv_expo_time;
      cv::Mat img_rgb = vio_manager->img_rgb;
      for (size_t i = 0; i < size; i++)
      {
        PointTypeRGB pointRGB;
        pointRGB.x = pcl_wait_pub->points[i].x;
        pointRGB.y = pcl_wait_pub->points[i].y;
        pointRGB.z = pcl_wait_pub->points[i].z;

        V3D p_w(pcl_wait_pub->points[i].x, pcl_wait_pub->points[i].y, pcl_wait_pub->points[i].z);
        V3D pf(vio_manager->new_frame_->w2f(p_w)); if (pf[2] < 0) continue;
        V2D pc(vio_manager->new_frame_->w2c(p_w));

        if (vio_manager->new_frame_->cam_->isInFrame(pc.cast<int>(), 3)) // 100
        {
          V3F pixel = vio_manager->getInterpolatedPixel(img_rgb, pc);
          pointRGB.r = pixel[2];
          pointRGB.g = pixel[1];
          pointRGB.b = pixel[0];
          // pointRGB.r = pixel[2] * inv_expo; pointRGB.g = pixel[1] * inv_expo; pointRGB.b = pixel[0] * inv_expo;
          // if (pointRGB.r > 255) pointRGB.r = 255;
          // else if (pointRGB.r < 0) pointRGB.r = 0;
          // if (pointRGB.g > 255) pointRGB.g = 255;
          // else if (pointRGB.g < 0) pointRGB.g = 0;
          // if (pointRGB.b > 255) pointRGB.b = 255;
          // else if (pointRGB.b < 0) pointRGB.b = 0;
          if (pf.norm() > blind_rgb_points) laserCloudWorldRGB->push_back(pointRGB);
        }
      }
    }
    else
    {
      pub_num++;
    }
  }

  /*** Publish Frame ***/
  sensor_msgs::PointCloud2 laserCloudmsg;
  if (img_en)
  {
    // cout << "RGB pointcloud size: " << laserCloudWorldRGB->size() << endl;
    pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
  }
  else 
  { 
    pcl::toROSMsg(*pcl_w_wait_pub, laserCloudmsg); 
  }
  laserCloudmsg.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
  laserCloudmsg.header.frame_id = "camera_init";
  pubLaserCloudFullRes.publish(laserCloudmsg);

  /**************** save map ****************/
  /* 1. make sure you have enough memories
  /* 2. noted that pcd save will influence the real-time performences **/
  if (pcd_save_en)
  {
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
    static int scan_wait_num = 0;

    if (img_en)
    {
      *pcl_wait_save += *laserCloudWorldRGB;
    }
    else
    {
      *pcl_wait_save_intensity += *pcl_w_wait_pub;
    }
    scan_wait_num++;

    if ((pcl_wait_save->size() > 0 || pcl_wait_save_intensity->size() > 0) && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
    {
      pcd_index++;
      string all_points_dir(string(string(ROOT_DIR) + "Log/PCD/") + to_string(pcd_index) + string(".pcd"));
      pcl::PCDWriter pcd_writer;
      if (pcd_save_en)
      {
        cout << "current scan saved to /PCD/" << all_points_dir << endl;
        if (img_en)
        {
          pcd_writer.writeBinary(all_points_dir, *pcl_wait_save); // pcl::io::savePCDFileASCII(all_points_dir, *pcl_wait_save);
          PointCloudXYZRGB().swap(*pcl_wait_save);
        }
        else
        {
          pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_intensity);
          PointCloudXYZI().swap(*pcl_wait_save_intensity);
        }        
        Eigen::Quaterniond q(_state.rot_end);
        fout_pcd_pos << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " " << q.w() << " " << q.x() << " " << q.y()
                     << " " << q.z() << " " << endl;
        scan_wait_num = 0;
      }
    }
  }
  if(laserCloudWorldRGB->size() > 0)  PointCloudXYZI().swap(*pcl_wait_pub); 
  PointCloudXYZI().swap(*pcl_w_wait_pub);
}

void LIVMapper::publish_visual_sub_map(const ros::Publisher &pubSubVisualMap)
{
  PointCloudXYZI::Ptr laserCloudFullRes(visual_sub_map);
  PointCloudXYZI::Ptr sub_pcl_visual_map_pub(new PointCloudXYZI());
  *sub_pcl_visual_map_pub = *laserCloudFullRes;
  if (1)
  {
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*sub_pcl_visual_map_pub, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time::now();
    laserCloudmsg.header.frame_id = "camera_init";
    pubSubVisualMap.publish(laserCloudmsg);
  }
}

void LIVMapper::publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list)
{
  int effect_feat_num = ptpl_list.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effect_feat_num, 1));
  for (int i = 0; i < effect_feat_num; i++)
  {
    laserCloudWorld->points[i].x = ptpl_list[i].point_w_[0];
    laserCloudWorld->points[i].y = ptpl_list[i].point_w_[1];
    laserCloudWorld->points[i].z = ptpl_list[i].point_w_[2];
  }
  sensor_msgs::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = ros::Time::now();
  laserCloudFullRes3.header.frame_id = "camera_init";
  pubLaserCloudEffect.publish(laserCloudFullRes3);
}

template <typename T> void LIVMapper::set_posestamp(T &out)
{
  out.position.x = _state.pos_end(0);
  out.position.y = _state.pos_end(1);
  out.position.z = _state.pos_end(2);
  out.orientation.x = geoQuat.x;
  out.orientation.y = geoQuat.y;
  out.orientation.z = geoQuat.z;
  out.orientation.w = geoQuat.w;
}

ros::Time LIVMapper::estimator_stamp() const
{
  // _state is the estimate at this sensor epoch, not at callback completion.
  // Stamping it with ros::Time::now() made replay latency look like motion and
  // produced run-dependent 30--240 ms trajectory offsets.  Preserve `now` only
  // as a defensive startup fallback before the first synchronized measurement.
  ros::Time stamp = ros::Time::now();
  if (std::isfinite(LidarMeasures.last_lio_update_time) &&
      LidarMeasures.last_lio_update_time > 0.0)
    stamp.fromSec(LidarMeasures.last_lio_update_time);
  return stamp;
}

void LIVMapper::publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id = "aft_mapped";
  odomAftMapped.header.stamp = estimator_stamp();
  set_posestamp(odomAftMapped.pose.pose);

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(_state.pos_end(0), _state.pos_end(1), _state.pos_end(2)));
  q.setW(geoQuat.w);
  q.setX(geoQuat.x);
  q.setY(geoQuat.y);
  q.setZ(geoQuat.z);
  transform.setRotation(q);
  br.sendTransform( tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "aft_mapped") );
  pubOdomAftMapped.publish(odomAftMapped);
}

// Hand-eye body publishing (body_calib path). Composes the published IMU pose with the FACTORY
// camera<->IMU extrinsic (vio_manager->Rci/Pci) and the calibrated camera->body extrinsic (config
// body_calib) to recover the body(=marker) pose. Two outputs:
//   /aft_mapped_to_body       PoseStamped, body in LIVO world W_L. Always (no GT needed). MUX vio input.
//   /aft_mapped_to_optitrack  Odometry, body in OptiTrack global. Needs GT; anchored once from a
//                             gt_avg_sec average so any deviation from /vision_pose/mocap is pure drift.
void LIVMapper::publish_body_optitrack()
{
  // H_{world<-body} = T_WI * inv(T_CI) * inv(T_BC)
  Eigen::Isometry3d T_WI = Eigen::Isometry3d::Identity();
  T_WI.linear() = _state.rot_end;  T_WI.translation() = _state.pos_end;            // IMU in world
  Eigen::Isometry3d T_CI = Eigen::Isometry3d::Identity();
  T_CI.linear() = vio_manager->Rci; T_CI.translation() = vio_manager->Pci;          // p_cam = Rci p_imu + Pci
  Eigen::Isometry3d T_BC = Eigen::Isometry3d::Identity();
  T_BC.linear() = R_cam2body;      T_BC.translation() = t_cam2body;                 // p_body = R p_cam + t
  Eigen::Isometry3d T_WB = T_WI * T_CI.inverse() * T_BC.inverse();

  const ros::Time now = estimator_stamp();
  // /aft_mapped_to_body lives in the ROS-oriented W_L frame. T_WB is in the OPTICAL
  // camera_init world (gravity_align off), so left-multiply the optical->ROS world flip
  // q_o2r=(x,y,z,w)=(0.5,-0.5,0.5,-0.5) — the same bridge the old /aft_mapped_to_odom
  // used. Without it optical +Z(forward) reads as ROS +Z and the body points up.
  // Unanchored on purpose (W_L origin; mocap-source offset jump is intended).
  static const Eigen::Quaterniond q_o2r(-0.5, 0.5, -0.5, 0.5);   // (w,x,y,z)
  Eigen::Isometry3d T_ros = Eigen::Isometry3d::Identity();
  T_ros.linear() = q_o2r.toRotationMatrix();
  Eigen::Isometry3d T_WB_ros = T_ros * T_WB;
  {
    Eigen::Quaterniond q(T_WB_ros.linear());
    geometry_msgs::PoseStamped pb;
    pb.header.stamp = now; pb.header.frame_id = "odom";   // unify with control-facing fast_livo topics (imu_propagate)
    pb.pose.position.x = T_WB_ros.translation().x();
    pb.pose.position.y = T_WB_ros.translation().y();
    pb.pose.position.z = T_WB_ros.translation().z();
    pb.pose.orientation.x = q.x(); pb.pose.orientation.y = q.y();
    pb.pose.orientation.z = q.z(); pb.pose.orientation.w = q.w();
    pubOdomAftMappedBody.publish(pb);
  }

  if (gravity_align_en && !gravity_align_finished) return;   // anchor only after gravity align settles
  if (gt_odom_received && !body_anchor_latched &&
      gt_buf.size() >= 2 && (gt_buf.back().first - gt_buf.front().first) >= gt_avg_sec) {
    Eigen::Vector3d t_avg = Eigen::Vector3d::Zero();
    Eigen::Vector4d q_acc = Eigen::Vector4d::Zero();
    Eigen::Quaterniond q0(gt_buf.front().second.linear());
    for (auto &kv : gt_buf) {
      t_avg += kv.second.translation();
      Eigen::Quaterniond qi(kv.second.linear());
      if (qi.dot(q0) < 0) qi.coeffs() *= -1.0;               // sign-align before averaging
      q_acc += qi.coeffs();
    }
    t_avg /= static_cast<double>(gt_buf.size());
    Eigen::Quaterniond q_avg; q_avg.coeffs() = q_acc.normalized();
    Eigen::Isometry3d T_GB = Eigen::Isometry3d::Identity();
    T_GB.linear() = q_avg.toRotationMatrix(); T_GB.translation() = t_avg;
    T_anchor = T_GB * T_WB.inverse();                        // optitrack <- W_L, constant
    body_anchor_latched = true;
    ROS_INFO("[body_calib] optitrack anchor latched from %.1fs GT avg (%zu samples)", gt_avg_sec, gt_buf.size());
  }
  if (!body_anchor_latched) return;                          // no global frame yet -> only /aft_mapped_to_body

  Eigen::Isometry3d T_GB = T_anchor * T_WB;
  Eigen::Quaterniond q(T_GB.linear());
  nav_msgs::Odometry od;
  od.header.stamp = now; od.header.frame_id = "optitrack"; od.child_frame_id = "body";
  od.pose.pose.position.x = T_GB.translation().x();
  od.pose.pose.position.y = T_GB.translation().y();
  od.pose.pose.position.z = T_GB.translation().z();
  od.pose.pose.orientation.x = q.x(); od.pose.pose.orientation.y = q.y();
  od.pose.pose.orientation.z = q.z(); od.pose.pose.orientation.w = q.w();
  pubOdomAftMappedOdom.publish(od);
}

// Mocap gt-init (ported from sim `ml`): the FIRST pose on gt_pose_topic defines
// odom -> camera_init. The internal EKF state is NOT changed — it stays at the
// camera_init origin (so /aft_mapped_to_init is still 0,0,0 at boot); instead the
// imu_prop odom publisher multiplies by this latched transform so its output starts
// at the true mocap pose.
void LIVMapper::gt_odom_cbk(const geometry_msgs::PoseStamped::ConstPtr &msg_in)
{
  // Legacy first-message latch (kept for the imu_prop odom path / non-body fallback).
  if (!gt_odom_received) {
    odom_to_camera_init.translation.x = msg_in->pose.position.x;
    odom_to_camera_init.translation.y = msg_in->pose.position.y;
    odom_to_camera_init.translation.z = msg_in->pose.position.z;
    odom_to_camera_init.rotation = msg_in->pose.orientation;
    gt_odom_received = true;
    ROS_INFO("[mocap] latched odom->camera_init from gt pose [%.3f, %.3f, %.3f]",
             odom_to_camera_init.translation.x, odom_to_camera_init.translation.y,
             odom_to_camera_init.translation.z);
  }
  // body_calib anchor: buffer GT (optitrack<-body) poses until latched, for a gt_avg_sec average
  // (single t=0 sample is ~1deg noisy -> cm-level constant offset over the whole trajectory).
  if (body_calib_en && !body_anchor_latched) {
    const auto &o = msg_in->pose.orientation; const auto &p = msg_in->pose.position;
    Eigen::Isometry3d g = Eigen::Isometry3d::Identity();
    g.linear() = Eigen::Quaterniond(o.w, o.x, o.y, o.z).normalized().toRotationMatrix();
    g.translation() = Eigen::Vector3d(p.x, p.y, p.z);
    gt_buf.emplace_back(msg_in->header.stamp.toSec(), g);
    while (gt_buf.size() > 2 && gt_buf.back().first - gt_buf.front().first > gt_avg_sec + 1.0)
      gt_buf.erase(gt_buf.begin());
  }
}

// Re-anchor on demand: reset LIVO state and re-latch on the next gt pose. Trigger with
//   rostopic pub -1 /livo/reinit std_msgs/Empty {}
// (sim `ml` does this via dynamic_reconfigure reinitialize_with_gt_odom; an Empty topic
// is the same reset with no extra build dependency).
void LIVMapper::reinit_cbk(const std_msgs::Empty::ConstPtr &msg_in)
{
  ROS_WARN("[mocap] reinit requested: resetting LIVO state, re-latch on next gt pose");
  gt_odom_received = false;
  body_anchor_latched = false;
  gt_buf.clear();
  p_imu->Reset();
  _state.resetpose();
  vio_manager->resetGrid();
  vio_manager->visual_submap->reset();
  lidar_map_inited = false;
  ekf_finish_once = false;
  is_first_frame = true;
}

void LIVMapper::publish_path(const ros::Publisher pubPath)
{
  set_posestamp(msg_body_pose.pose);
  msg_body_pose.header.stamp = estimator_stamp();
  msg_body_pose.header.frame_id = "camera_init";
  path.poses.push_back(msg_body_pose);
  pubPath.publish(path);
}
