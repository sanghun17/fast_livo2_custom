/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "vio.h"

#include <filesystem>

namespace
{
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix7d = Eigen::Matrix<double, 7, 7>;

constexpr double kMinCameraDepthM = 1e-6;

bool hasFinitePositiveDepth(const V3D &point_camera)
{
  return point_camera.allFinite() && point_camera[2] > kMinCameraDepthM;
}

bool voxelLocationLess(const VOXEL_LOCATION &lhs, const VOXEL_LOCATION &rhs)
{
  if (lhs.x != rhs.x) return lhs.x < rhs.x;
  if (lhs.y != rhs.y) return lhs.y < rhs.y;
  return lhs.z < rhs.z;
}

bool pointPositionLess(const V3D &lhs, const V3D &rhs)
{
  if (lhs.x() != rhs.x()) return lhs.x() < rhs.x();
  if (lhs.y() != rhs.y()) return lhs.y() < rhs.y();
  return lhs.z() < rhs.z();
}

bool preferMapPoint(float candidate_distance, const VisualPoint *candidate,
                    float selected_distance, const VisualPoint *selected)
{
  if (selected == nullptr || candidate_distance < selected_distance) return true;
  return candidate_distance == selected_distance &&
         pointPositionLess(candidate->pos_, selected->pos_);
}

bool preferPointCandidate(float candidate_score, const pointWithVar &candidate,
                          int selected_type, float selected_score,
                          const pointWithVar &selected)
{
  if (!std::isfinite(candidate_score)) return false;
  if (candidate_score > selected_score) return true;
  // Preserve the upstream threshold semantics: a zero-score candidate does
  // not populate an empty grid whose reset score is also zero.
  return selected_type == VIOManager::TYPE_POINTCLOUD &&
         candidate_score == selected_score &&
         pointPositionLess(candidate.point_w, selected.point_w);
}

bool patchSamplingInBounds(const V2D &pixel, int scale, int patch_size,
                           int patch_size_half, const cv::Mat &image,
                           int expected_width, int expected_height,
                           bool needs_gradient_border)
{
  if (!pixel.allFinite() || scale <= 0 || patch_size <= 0 || image.empty() ||
      image.cols != expected_width || image.rows != expected_height ||
      image.type() != CV_8UC1 || !image.isContinuous())
    return false;

  const int u_anchor = static_cast<int>(std::floor(pixel[0] / scale)) * scale;
  const int v_anchor = static_cast<int>(std::floor(pixel[1] / scale)) * scale;
  const int lower_extra = needs_gradient_border ? scale : 0;
  const int upper_extra = needs_gradient_border ? 2 * scale : scale;
  const int min_u = u_anchor - patch_size_half * scale - lower_extra;
  const int min_v = v_anchor - patch_size_half * scale - lower_extra;
  const int max_u = u_anchor + (patch_size - 1 - patch_size_half) * scale +
                    upper_extra;
  const int max_v = v_anchor + (patch_size - 1 - patch_size_half) * scale +
                    upper_extra;
  return min_u >= 0 && min_v >= 0 && max_u < image.cols &&
         max_v < image.rows;
}

int checkedGridIndex(const V2D &pixel, int grid_size, int grid_n_width,
                     int grid_n_height)
{
  if (!pixel.allFinite() || grid_size <= 0 || grid_n_width <= 0 ||
      grid_n_height <= 0)
    return -1;
  const int col = static_cast<int>(std::floor(pixel[0] / grid_size));
  const int row = static_cast<int>(std::floor(pixel[1] / grid_size));
  if (col < 0 || col >= grid_n_width || row < 0 || row >= grid_n_height)
    return -1;
  return row * grid_n_width + col;
}

struct VisualInfoMetrics
{
  double rot_trace = 0.0;
  double rot_eig_min = 0.0;
  double rot_eig_mid = 0.0;
  double rot_eig_max = 0.0;
  double rot_condition_ratio = 0.0;
  double trans_trace = 0.0;
  double trans_eig_min = 0.0;
  double trans_eig_mid = 0.0;
  double trans_eig_max = 0.0;
  double trans_condition_ratio = 0.0;
  double pose_trace = 0.0;
  double pose_eig_min = 0.0;
  double pose_eig_max = 0.0;
  double pose_condition_ratio = 0.0;
};

template <int N>
Eigen::Matrix<double, N, 1> nonnegativeEigenvalues(
    const Eigen::Matrix<double, N, N> &matrix)
{
  const Eigen::Matrix<double, N, N> symmetric =
      0.5 * (matrix + matrix.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> solver(symmetric);
  Eigen::Matrix<double, N, 1> values = solver.eigenvalues();
  for (int i = 0; i < N; ++i) values[i] = std::max(0.0, values[i]);
  return values;
}

VisualInfoMetrics computeVisualInfoMetrics(const Matrix6d &raw_info)
{
  VisualInfoMetrics result;
  const Matrix6d info = 0.5 * (raw_info + raw_info.transpose());
  const M3D rot_info = info.block<3, 3>(0, 0);
  const M3D trans_info = info.block<3, 3>(3, 3);
  const V3D rot_eigs = nonnegativeEigenvalues<3>(rot_info);
  const V3D trans_eigs = nonnegativeEigenvalues<3>(trans_info);
  const Eigen::Matrix<double, 6, 1> pose_eigs =
      nonnegativeEigenvalues<6>(info);

  result.rot_trace = rot_info.trace();
  result.rot_eig_min = rot_eigs[0];
  result.rot_eig_mid = rot_eigs[1];
  result.rot_eig_max = rot_eigs[2];
  result.rot_condition_ratio = rot_eigs[0] / std::max(1e-12, rot_eigs[2]);
  result.trans_trace = trans_info.trace();
  result.trans_eig_min = trans_eigs[0];
  result.trans_eig_mid = trans_eigs[1];
  result.trans_eig_max = trans_eigs[2];
  result.trans_condition_ratio =
      trans_eigs[0] / std::max(1e-12, trans_eigs[2]);
  result.pose_trace = info.trace();
  result.pose_eig_min = pose_eigs[0];
  result.pose_eig_max = pose_eigs[5];
  result.pose_condition_ratio = pose_eigs[0] / std::max(1e-12, pose_eigs[5]);

  return result;
}

template <int N>
Eigen::Matrix<double, N, N> regularizedSPD(
    const Eigen::Matrix<double, N, N> &raw_matrix)
{
  const Eigen::Matrix<double, N, N> symmetric =
      0.5 * (raw_matrix + raw_matrix.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> solver(symmetric);
  Eigen::Matrix<double, N, 1> eigenvalues = solver.eigenvalues();
  const double scale = std::max(1.0, eigenvalues.cwiseAbs().maxCoeff());
  const double floor = 1e-12 * scale;
  for (int i = 0; i < N; ++i) eigenvalues[i] = std::max(floor, eigenvalues[i]);
  return solver.eigenvectors() * eigenvalues.asDiagonal() *
         solver.eigenvectors().transpose();
}

template <int N>
double stableLogDetSPD(const Eigen::Matrix<double, N, N> &matrix)
{
  const Eigen::Matrix<double, N, N> spd = regularizedSPD<N>(matrix);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> solver(spd);
  return solver.eigenvalues().array().log().sum();
}

Matrix6d exposureMarginalizedPoseInfo(const Matrix7d &raw_normal,
                                      bool exposure_estimate_enabled)
{
  const Matrix7d normal = 0.5 * (raw_normal + raw_normal.transpose());
  Matrix6d marginalized = normal.block<6, 6>(0, 0);
  if (exposure_estimate_enabled)
  {
    const double exposure_info = normal(6, 6);
    const double regularizer = std::max(1e-12, std::abs(exposure_info) * 1e-12);
    const Eigen::Matrix<double, 6, 1> cross = normal.block<6, 1>(0, 6);
    marginalized -= (cross * cross.transpose()) /
                    std::max(regularizer, exposure_info);
  }
  marginalized = 0.5 * (marginalized + marginalized.transpose());
  // Schur subtraction can produce tiny negative eigenvalues through roundoff.
  // Project only the diagnostic copy to PSD; estimator matrices are untouched.
  Eigen::SelfAdjointEigenSolver<Matrix6d> solver(marginalized);
  Eigen::Matrix<double, 6, 1> eigenvalues = solver.eigenvalues();
  for (int i = 0; i < 6; ++i) eigenvalues[i] = std::max(0.0, eigenvalues[i]);
  return solver.eigenvectors() * eigenvalues.asDiagonal() *
         solver.eigenvectors().transpose();
}

double poseInformationGainNats(const Matrix7d &raw_prior_cov,
                               const Matrix7d &raw_measurement_normal)
{
  // Exposure is a shared nuisance state.  Incorporate the complete 7x7 prior,
  // including pose/exposure correlation, update it with the complete normal
  // matrix, and compare the marginalized 6x6 pose covariance determinants.
  const Matrix7d prior_cov = regularizedSPD<7>(raw_prior_cov);
  const Matrix7d prior_precision = prior_cov.inverse();
  const Matrix7d posterior_cov =
      regularizedSPD<7>(prior_precision + raw_measurement_normal).inverse();
  const Matrix6d prior_pose_cov = prior_cov.block<6, 6>(0, 0);
  const Matrix6d posterior_pose_cov = posterior_cov.block<6, 6>(0, 0);
  const double gain = 0.5 *
      (stableLogDetSPD<6>(prior_pose_cov) -
       stableLogDetSPD<6>(posterior_pose_cov));
  return std::max(0.0, gain);
}
}  // namespace

VIOManager::VIOManager()
{
  // downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
}

VIOManager::~VIOManager()
{
  if (visual_quality_frames_stream.is_open()) visual_quality_frames_stream.close();
  if (visual_quality_points_stream.is_open()) visual_quality_points_stream.close();
  delete visual_submap;
  for (auto& pair : warp_map) delete pair.second;
  warp_map.clear();
  for (auto& pair : feat_map) delete pair.second;
  feat_map.clear();
}

bool VIOManager::ensureVisualQualityLogOpen()
{
  if (!visual_quality_log_enabled) return false;
  if (visual_quality_frames_stream.is_open() &&
      visual_quality_points_stream.is_open()) return true;

  try
  {
    const std::filesystem::path prefix_path(visual_quality_output_prefix);
    if (!prefix_path.parent_path().empty())
      std::filesystem::create_directories(prefix_path.parent_path());
  }
  catch (const std::exception &error)
  {
    std::cerr << "[VIO quality] cannot create output directory for '"
              << visual_quality_output_prefix << "': " << error.what() << std::endl;
    visual_quality_log_enabled = false;
    return false;
  }

  visual_quality_frames_stream.open(
      visual_quality_output_prefix + "_frames.csv", std::ios::out | std::ios::trunc);
  visual_quality_points_stream.open(
      visual_quality_output_prefix + "_points.csv", std::ios::out | std::ios::trunc);
  if (!visual_quality_frames_stream.good() || !visual_quality_points_stream.good())
  {
    std::cerr << "[VIO quality] cannot open CSV prefix '"
              << visual_quality_output_prefix << "'" << std::endl;
    if (visual_quality_frames_stream.is_open()) visual_quality_frames_stream.close();
    if (visual_quality_points_stream.is_open()) visual_quality_points_stream.close();
    visual_quality_log_enabled = false;
    return false;
  }

  visual_quality_frames_stream
      << "frame_index,img_time_s,img_rel_s,width,height,state_update_enabled,"
      << "active_count,valid_final_count,improved_count,improved_ratio,"
      << "prop_sse_sum,final_sse_sum,error_ratio,rot_trace,rot_eig_min,"
      << "rot_eig_mid,rot_eig_max,rot_condition_ratio,trans_trace,"
      << "trans_eig_min,trans_eig_mid,trans_eig_max,trans_condition_ratio,"
      << "pose_trace,pose_eig_min,pose_eig_max,pose_condition_ratio,"
      << "marg_rot_trace,marg_rot_eig_min,marg_rot_eig_mid,marg_rot_eig_max,"
      << "marg_rot_condition_ratio,marg_trans_trace,marg_trans_eig_min,"
      << "marg_trans_eig_mid,marg_trans_eig_max,marg_trans_condition_ratio,"
      << "marg_pose_trace,marg_pose_eig_min,marg_pose_eig_max,"
      << "marg_pose_condition_ratio,pose_information_gain_nats\n";
  visual_quality_points_stream
      << "frame_index,img_time_s,img_rel_s,point_index,stage,u,v,x_w,y_w,z_w,"
      << "depth_m,range_m,view_cos,shi_tomasi,ncc,search_level,prop_sse,"
      << "retrieve_prop_sse,prop_rmse,final_sse,final_rmse,improved,rot_trace,rot_eig_min,"
      << "rot_eig_mid,rot_eig_max,rot_condition_ratio,trans_trace,"
      << "trans_eig_min,trans_eig_mid,trans_eig_max,trans_condition_ratio,"
      << "pose_trace,pose_eig_min,pose_eig_max,pose_condition_ratio,"
      << "marg_rot_trace,marg_rot_eig_min,marg_rot_eig_mid,marg_rot_eig_max,"
      << "marg_rot_condition_ratio,marg_trans_trace,marg_trans_eig_min,"
      << "marg_trans_eig_mid,marg_trans_eig_max,marg_trans_condition_ratio,"
      << "marg_pose_trace,marg_pose_eig_min,marg_pose_eig_max,"
      << "marg_pose_condition_ratio\n";
  visual_quality_frames_stream << std::setprecision(17);
  visual_quality_points_stream << std::setprecision(17);
  std::cout << "[VIO quality] logging frames and active patches to '"
            << visual_quality_output_prefix << "_{frames,points}.csv'" << std::endl;
  return true;
}

void VIOManager::logVisualQualityFrame(const cv::Mat &img,
                                       double img_time_s,
                                       double img_rel_s)
{
  if (!ensureVisualQualityLogOpen()) return;
  if (inverse_composition_en)
  {
    static bool warned = false;
    if (!warned)
    {
      std::cerr << "[VIO quality] disabled: diagnostics currently mirror the "
                << "forward updateState path and require "
                << "vio/inverse_composition_en=false" << std::endl;
      warned = true;
    }
    visual_quality_log_enabled = false;
    return;
  }

  const std::uint64_t frame_index = visual_quality_frame_index++;
  const int active_count = static_cast<int>(visual_submap->voxel_points.size());
  const double measurement_variance = std::max(1e-12, img_point_cov);
  // Snapshot taken immediately before the VIO update in processFrame().  Keep
  // this distinct from state->cov here, which is already the VIO posterior.
  const Matrix7d prior_state_cov = visual_quality_prior_state_cov;
  Matrix7d frame_normal = Matrix7d::Zero();
  double propagated_sse_sum = 0.0;
  double final_sse_sum = 0.0;
  int valid_final_count = 0;
  int improved_count = 0;
  const double nan = std::numeric_limits<double>::quiet_NaN();

  for (int i = 0; i < active_count; ++i)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];
    const double retrieve_propagated_sse =
        i < static_cast<int>(visual_submap->propa_errors.size())
            ? static_cast<double>(visual_submap->propa_errors[i]) : nan;
    const int search_level =
        i < static_cast<int>(visual_submap->search_levels.size())
            ? visual_submap->search_levels[i] : -1;
    const char *stage = state_update_enabled ? "accepted_active" : "active_tracking_only";

    if (pt == nullptr)
    {
      visual_quality_points_stream
          << frame_index << ',' << img_time_s << ',' << img_rel_s << ',' << i
          << ",null_visual_point";
      for (int column = 5; column < 50; ++column)
        visual_quality_points_stream << ',' << nan;
      visual_quality_points_stream << '\n';
      continue;
    }

    const V3D point_camera = new_frame_->w2f(pt->pos_);
    const bool valid_search_level = search_level >= 0 && search_level < 30;
    const int scale = valid_search_level ? (1 << search_level) : 1;
    const V2D pixel = hasFinitePositiveDepth(point_camera)
                          ? cam->world2cam(point_camera)
                          : V2D::Constant(nan);
    const int diagnostic_border = (patch_size_half + 1) * scale;
    const bool valid_projection = valid_search_level &&
        hasFinitePositiveDepth(point_camera) &&
        patchSamplingInBounds(pixel, scale, patch_size, patch_size_half, img,
                              width, height, true);
    if (!valid_projection ||
        i >= static_cast<int>(visual_submap->warp_patch.size()) ||
        i >= static_cast<int>(visual_submap->inv_expo_list.size()) ||
        visual_submap->warp_patch[i].size() <
            static_cast<size_t>(patch_size_total))
    {
      visual_quality_points_stream
          << frame_index << ',' << img_time_s << ',' << img_rel_s << ',' << i
          << ",invalid_final_projection," << pixel[0] << ',' << pixel[1]
          << ',' << pt->pos_[0] << ',' << pt->pos_[1] << ',' << pt->pos_[2]
          << ',' << point_camera[2] << ',' << point_camera.norm() << ',' << nan
          << ',' << nan << ',' << nan << ',' << search_level
          << ',' << nan << ',' << retrieve_propagated_sse;
      for (int column = 18; column < 50; ++column)
        visual_quality_points_stream << ',' << nan;
      visual_quality_points_stream << '\n';
      continue;
    }

    MD(2, 3) projection_jacobian;
    computeProjectionJacobian(point_camera, projection_jacobian);
    M3D point_hat;
    point_hat << SKEW_SYM_MATRX(point_camera);
    const M3D Rwi(state->rot_end);
    const M3D Jdp_dt_local = Rci * Rwi.transpose();
    const double u = pixel[0];
    const double v = pixel[1];
    const int u_i = floorf(pixel[0] / scale) * scale;
    const int v_i = floorf(pixel[1] / scale) * scale;
    const float subpixel_u = static_cast<float>((u - u_i) / scale);
    const float subpixel_v = static_cast<float>((v - v_i) / scale);
    const float weight_tl = (1.0f - subpixel_u) * (1.0f - subpixel_v);
    const float weight_tr = subpixel_u * (1.0f - subpixel_v);
    const float weight_bl = (1.0f - subpixel_u) * subpixel_v;
    const float weight_br = subpixel_u * subpixel_v;
    const std::vector<float> &reference_patch = visual_submap->warp_patch[i];
    const double inverse_reference_exposure = visual_submap->inv_expo_list[i];
    // Recompute the propagated error on the same search-level sampling grid as
    // the final finest-level update.  The upstream propa_errors value is kept
    // separately because it samples the current image at unit scale, making
    // its green/blue comparison inconsistent when search_level > 0.
    const auto compute_sse_at_state = [&](const StatesGroup &sample_state) {
      const M3D sample_Rcw = Rci * sample_state.rot_end.transpose();
      const V3D sample_Pcw = -Rci * sample_state.rot_end.transpose() *
                             sample_state.pos_end + Pci;
      const V3D sample_point_camera = sample_Rcw * pt->pos_ + sample_Pcw;
      if (!hasFinitePositiveDepth(sample_point_camera)) return nan;
      const V2D sample_pixel = cam->world2cam(sample_point_camera);
      if (!patchSamplingInBounds(sample_pixel, scale, patch_size,
                                 patch_size_half, img, width, height, false))
        return nan;
      const int sample_u_i = floorf(sample_pixel[0] / scale) * scale;
      const int sample_v_i = floorf(sample_pixel[1] / scale) * scale;
      const float sample_subpixel_u =
          static_cast<float>((sample_pixel[0] - sample_u_i) / scale);
      const float sample_subpixel_v =
          static_cast<float>((sample_pixel[1] - sample_v_i) / scale);
      const float sample_weight_tl =
          (1.0f - sample_subpixel_u) * (1.0f - sample_subpixel_v);
      const float sample_weight_tr = sample_subpixel_u * (1.0f - sample_subpixel_v);
      const float sample_weight_bl = (1.0f - sample_subpixel_u) * sample_subpixel_v;
      const float sample_weight_br = sample_subpixel_u * sample_subpixel_v;
      double sse = 0.0;
      for (int x = 0; x < patch_size; ++x)
      {
        uint8_t *sample_pointer = (uint8_t *)img.data +
            (sample_v_i + x * scale - patch_size_half * scale) * width +
            sample_u_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, sample_pointer += scale)
        {
          const int patch_index = x * patch_size + y;
          const double current_value =
              sample_weight_tl * sample_pointer[0] +
              sample_weight_tr * sample_pointer[scale] +
              sample_weight_bl * sample_pointer[scale * width] +
              sample_weight_br * sample_pointer[scale * width + scale];
          const double residual = sample_state.inv_expo_time * current_value -
              inverse_reference_exposure * reference_patch[patch_index];
          sse += residual * residual;
        }
      }
      return sse;
    };
    const double propagated_sse =
        compute_sse_at_state(visual_quality_prior_state);
    std::vector<float> current_patch(patch_size_total, 0.0f);
    Eigen::Matrix<double, Eigen::Dynamic, 7> patch_jacobian(patch_size_total, 7);
    patch_jacobian.setZero();
    double final_sse = 0.0;

    for (int x = 0; x < patch_size; ++x)
    {
      uint8_t *image_pointer = (uint8_t *)img.data +
          (v_i + x * scale - patch_size_half * scale) * width +
          u_i - patch_size_half * scale;
      for (int y = 0; y < patch_size; ++y, image_pointer += scale)
      {
        const float du = 0.5f *
            ((weight_tl * image_pointer[scale] + weight_tr * image_pointer[2 * scale] +
              weight_bl * image_pointer[scale * width + scale] +
              weight_br * image_pointer[scale * width + 2 * scale]) -
             (weight_tl * image_pointer[-scale] + weight_tr * image_pointer[0] +
              weight_bl * image_pointer[scale * width - scale] +
              weight_br * image_pointer[scale * width]));
        const float dv = 0.5f *
            ((weight_tl * image_pointer[scale * width] +
              weight_tr * image_pointer[scale * width + scale] +
              weight_bl * image_pointer[2 * scale * width] +
              weight_br * image_pointer[2 * scale * width + scale]) -
             (weight_tl * image_pointer[-scale * width] +
              weight_tr * image_pointer[-scale * width + scale] +
              weight_bl * image_pointer[0] + weight_br * image_pointer[scale]));
        MD(1, 2) image_jacobian;
        image_jacobian << du, dv;
        image_jacobian *= state->inv_expo_time;
        image_jacobian *= 1.0 / scale;
        const MD(1, 3) Jdphi = image_jacobian * projection_jacobian * point_hat;
        const MD(1, 3) Jdp = -image_jacobian * projection_jacobian;
        const MD(1, 3) JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
        const MD(1, 3) Jdt = Jdp * Jdp_dt_local;
        const int patch_index = x * patch_size + y;
        const double current_value =
            weight_tl * image_pointer[0] + weight_tr * image_pointer[scale] +
            weight_bl * image_pointer[scale * width] +
            weight_br * image_pointer[scale * width + scale];
        if (exposure_estimate_en)
          patch_jacobian.block<1, 7>(patch_index, 0) << JdR, Jdt, current_value;
        else
        {
          patch_jacobian.block<1, 6>(patch_index, 0) << JdR, Jdt;
          patch_jacobian(patch_index, 6) = 0.0;
        }
        current_patch[patch_index] = static_cast<float>(current_value);
        const double residual = state->inv_expo_time * current_value -
            inverse_reference_exposure * reference_patch[patch_index];
        final_sse += residual * residual;
      }
    }

    const Matrix7d patch_normal =
        (patch_jacobian.transpose() * patch_jacobian) / measurement_variance;
    const Matrix6d raw_patch_pose_info = patch_normal.block<6, 6>(0, 0);
    const Matrix6d marginalized_patch_pose_info =
        exposureMarginalizedPoseInfo(patch_normal, exposure_estimate_en);
    const VisualInfoMetrics metrics = computeVisualInfoMetrics(raw_patch_pose_info);
    const VisualInfoMetrics marginalized_metrics =
        computeVisualInfoMetrics(marginalized_patch_pose_info);
    frame_normal += patch_normal;
    ++valid_final_count;
    if (std::isfinite(propagated_sse)) propagated_sse_sum += propagated_sse;
    final_sse_sum += final_sse;
    const bool improved = std::isfinite(propagated_sse) && final_sse <= propagated_sse;
    if (improved) ++improved_count;
    const double propagated_rmse = std::isfinite(propagated_sse)
        ? std::sqrt(propagated_sse / std::max(1, patch_size_total)) : nan;
    const double final_rmse = std::sqrt(final_sse / std::max(1, patch_size_total));
    const double shi_tomasi = vk::shiTomasiScore(img, pixel[0], pixel[1]);
    const double ncc = calculateNCC(const_cast<float *>(reference_patch.data()),
                                    current_patch.data(), patch_size_total);
    const V3D normal_camera = new_frame_->T_f_w_.rotation_matrix() * pt->normal_;
    const double view_cos = normal_camera.norm() > 1e-12 && point_camera.norm() > 1e-12
        ? std::abs(normal_camera.normalized().dot(point_camera.normalized())) : nan;

    visual_quality_points_stream
        << frame_index << ',' << img_time_s << ',' << img_rel_s << ',' << i
        << ',' << stage << ',' << u << ',' << v
        << ',' << pt->pos_[0] << ',' << pt->pos_[1] << ',' << pt->pos_[2]
        << ',' << point_camera[2] << ',' << point_camera.norm() << ',' << view_cos
        << ',' << shi_tomasi << ',' << ncc << ',' << search_level
        << ',' << propagated_sse << ',' << retrieve_propagated_sse
        << ',' << propagated_rmse
        << ',' << final_sse << ',' << final_rmse << ',' << (improved ? 1 : 0)
        << ',' << metrics.rot_trace << ',' << metrics.rot_eig_min
        << ',' << metrics.rot_eig_mid << ',' << metrics.rot_eig_max
        << ',' << metrics.rot_condition_ratio << ',' << metrics.trans_trace
        << ',' << metrics.trans_eig_min << ',' << metrics.trans_eig_mid
        << ',' << metrics.trans_eig_max << ',' << metrics.trans_condition_ratio
        << ',' << metrics.pose_trace << ',' << metrics.pose_eig_min
        << ',' << metrics.pose_eig_max << ',' << metrics.pose_condition_ratio
        << ',' << marginalized_metrics.rot_trace
        << ',' << marginalized_metrics.rot_eig_min
        << ',' << marginalized_metrics.rot_eig_mid
        << ',' << marginalized_metrics.rot_eig_max
        << ',' << marginalized_metrics.rot_condition_ratio
        << ',' << marginalized_metrics.trans_trace
        << ',' << marginalized_metrics.trans_eig_min
        << ',' << marginalized_metrics.trans_eig_mid
        << ',' << marginalized_metrics.trans_eig_max
        << ',' << marginalized_metrics.trans_condition_ratio
        << ',' << marginalized_metrics.pose_trace
        << ',' << marginalized_metrics.pose_eig_min
        << ',' << marginalized_metrics.pose_eig_max
        << ',' << marginalized_metrics.pose_condition_ratio << '\n';
  }

  const Matrix6d frame_pose_info = frame_normal.block<6, 6>(0, 0);
  const Matrix6d marginalized_frame_pose_info =
      exposureMarginalizedPoseInfo(frame_normal, exposure_estimate_en);
  const VisualInfoMetrics frame_metrics = computeVisualInfoMetrics(frame_pose_info);
  const VisualInfoMetrics marginalized_frame_metrics =
      computeVisualInfoMetrics(marginalized_frame_pose_info);
  const double pose_information_gain_nats =
      poseInformationGainNats(prior_state_cov, frame_normal);
  const double improved_ratio = valid_final_count > 0
      ? static_cast<double>(improved_count) / valid_final_count : 0.0;
  const double error_ratio = propagated_sse_sum > 1e-12
      ? final_sse_sum / propagated_sse_sum : 1.0;
  visual_quality_frames_stream
      << frame_index << ',' << img_time_s << ',' << img_rel_s
      << ',' << img.cols << ',' << img.rows << ',' << (state_update_enabled ? 1 : 0)
      << ',' << active_count << ',' << valid_final_count << ',' << improved_count
      << ',' << improved_ratio << ',' << propagated_sse_sum << ',' << final_sse_sum
      << ',' << error_ratio << ',' << frame_metrics.rot_trace
      << ',' << frame_metrics.rot_eig_min << ',' << frame_metrics.rot_eig_mid
      << ',' << frame_metrics.rot_eig_max << ',' << frame_metrics.rot_condition_ratio
      << ',' << frame_metrics.trans_trace << ',' << frame_metrics.trans_eig_min
      << ',' << frame_metrics.trans_eig_mid << ',' << frame_metrics.trans_eig_max
      << ',' << frame_metrics.trans_condition_ratio << ',' << frame_metrics.pose_trace
      << ',' << frame_metrics.pose_eig_min << ',' << frame_metrics.pose_eig_max
      << ',' << frame_metrics.pose_condition_ratio
      << ',' << marginalized_frame_metrics.rot_trace
      << ',' << marginalized_frame_metrics.rot_eig_min
      << ',' << marginalized_frame_metrics.rot_eig_mid
      << ',' << marginalized_frame_metrics.rot_eig_max
      << ',' << marginalized_frame_metrics.rot_condition_ratio
      << ',' << marginalized_frame_metrics.trans_trace
      << ',' << marginalized_frame_metrics.trans_eig_min
      << ',' << marginalized_frame_metrics.trans_eig_mid
      << ',' << marginalized_frame_metrics.trans_eig_max
      << ',' << marginalized_frame_metrics.trans_condition_ratio
      << ',' << marginalized_frame_metrics.pose_trace
      << ',' << marginalized_frame_metrics.pose_eig_min
      << ',' << marginalized_frame_metrics.pose_eig_max
      << ',' << marginalized_frame_metrics.pose_condition_ratio
      << ',' << pose_information_gain_nats << '\n';

  if (visual_quality_flush_every_n_frames <= 1 ||
      visual_quality_frame_index % visual_quality_flush_every_n_frames == 0)
  {
    visual_quality_frames_stream.flush();
    visual_quality_points_stream.flush();
  }
}

void VIOManager::setImuToLidarExtrinsic(const V3D &transl, const M3D &rot)
{
  Pli = -rot.transpose() * transl;
  Rli = rot.transpose();
}

void VIOManager::setLidarToCameraExtrinsic(vector<double> &R, vector<double> &P)
{
  Rcl << MAT_FROM_ARRAY(R);
  Pcl << VEC_FROM_ARRAY(P);
}

void VIOManager::initializeVIO()
{
  visual_submap = new SubSparseMap;

  fx = cam->fx();
  fy = cam->fy();
  cx = cam->cx();
  cy = cam->cy();
  image_resize_factor = cam->scale();

  printf("intrinsic: %.6lf, %.6lf, %.6lf, %.6lf\n", fx, fy, cx, cy);

  width = cam->width();
  height = cam->height();

  if (width <= 0 || height <= 0)
    throw std::runtime_error("VIO camera dimensions must be positive");

  printf("width: %d, height: %d, scale: %f\n", width, height, image_resize_factor);
  Rci = Rcl * Rli;
  Pci = Rcl * Pli + Pcl;

  V3D Pic;
  M3D tmp;
  Jdphi_dR = Rci;
  Pic = -Rci.transpose() * Pci;
  tmp << SKEW_SYM_MATRX(Pic);
  Jdp_dR = -Rci * tmp;

  if (grid_size > 10)
  {
    grid_n_width = (width + grid_size - 1) / grid_size;
    grid_n_height = (height + grid_size - 1) / grid_size;
  }
  else
  {
    const int requested_grid_rows = std::max(1, grid_n_height);
    grid_size = std::max(1, height / requested_grid_rows);
    grid_n_height = (height + grid_size - 1) / grid_size;
    grid_n_width = (width + grid_size - 1) / grid_size;
  }
  length = grid_n_width * grid_n_height;

  if(raycast_en)
  {
    // cv::Mat img_test = cv::Mat::zeros(height, width, CV_8UC1);
    // uchar* it = (uchar*)img_test.data;

    border_flag.resize(length, 0);

    std::vector<std::vector<V3D>>().swap(rays_with_sample_points);
    rays_with_sample_points.reserve(length);
    printf("grid_size: %d, grid_n_height: %d, grid_n_width: %d, length: %d\n", grid_size, grid_n_height, grid_n_width, length);

    float d_min = 0.1;
    float d_max = 3.0;
    float step = 0.2;
    for (int grid_row = 1; grid_row <= grid_n_height; grid_row++)
    {
      for (int grid_col = 1; grid_col <= grid_n_width; grid_col++)
      {
        std::vector<V3D> SamplePointsEachGrid;
        int index = (grid_row - 1) * grid_n_width + grid_col - 1;

        if (grid_row == 1 || grid_col == 1 || grid_row == grid_n_height || grid_col == grid_n_width) border_flag[index] = 1;

        int u = std::min(width - 1, grid_size / 2 + (grid_col - 1) * grid_size);
        int v = std::min(height - 1, grid_size / 2 + (grid_row - 1) * grid_size);
        // it[ u + v * width ] = 255;
        for (float d_temp = d_min; d_temp <= d_max; d_temp += step)
        {
          V3D xyz;
          xyz = cam->cam2world(u, v);
          xyz *= d_temp / xyz[2];
          // xyz[0] = (u - cx) / fx * d_temp;
          // xyz[1] = (v - cy) / fy * d_temp;
          // xyz[2] = d_temp;
          SamplePointsEachGrid.push_back(xyz);
        }
        rays_with_sample_points.push_back(SamplePointsEachGrid);
      }
    }
    // printf("rays_with_sample_points: %d, RaysWithSamplePointsCapacity: %d,
    // rays_with_sample_points[0].capacity(): %d, rays_with_sample_points[0]: %d\n",
    // rays_with_sample_points.size(), rays_with_sample_points.capacity(),
    // rays_with_sample_points[0].capacity(), rays_with_sample_points[0].size()); for
    // (const auto & it : rays_with_sample_points[0]) cout << it.transpose() << endl;
    // cv::imshow("img_test", img_test);
    // cv::waitKey(1);
  }

  if(colmap_output_en)
  {
    pinhole_cam = dynamic_cast<vk::PinholeCamera*>(cam);
    fout_colmap.open(DEBUG_FILE_DIR("Colmap/sparse/0/images.txt"), ios::out);
    fout_colmap << "# Image list with two lines of data per image:\n";
    fout_colmap << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
    fout_colmap << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
    fout_camera.open(DEBUG_FILE_DIR("Colmap/sparse/0/cameras.txt"), ios::out);
    fout_camera << "# Camera list with one line of data per camera:\n";
    fout_camera << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
    fout_camera << "1 PINHOLE " << width << " " << height << " "
        << std::fixed << std::setprecision(6)  // 控制浮点数精度为10位
        << fx << " " << fy << " "
        << cx << " " << cy << std::endl;
    fout_camera.close();
  }
  grid_num.resize(length);
  map_index.resize(length);
  map_dist.resize(length);
  update_flag.resize(length);
  scan_value.resize(length);

  patch_size_total = patch_size * patch_size;
  patch_size_half = static_cast<int>(patch_size / 2);
  patch_buffer.resize(patch_size_total);
  warp_len = patch_size_total * patch_pyrimid_level;
  border = (patch_size_half + 1) * (1 << patch_pyrimid_level);

  retrieve_voxel_points.reserve(length);
  append_voxel_points.reserve(length);

  sub_feat_map.clear();
}

void VIOManager::resetGrid()
{
  fill(grid_num.begin(), grid_num.end(), TYPE_UNKNOWN);
  fill(map_index.begin(), map_index.end(), 0);
  fill(map_dist.begin(), map_dist.end(), 10000.0f);
  fill(update_flag.begin(), update_flag.end(), 0);
  fill(scan_value.begin(), scan_value.end(), 0.0f);

  retrieve_voxel_points.clear();
  retrieve_voxel_points.resize(length);

  append_voxel_points.clear();
  append_voxel_points.resize(length);

  total_points = 0;
}

// void VIOManager::resetRvizDisplay()
// {
  // sub_map_ray.clear();
  // sub_map_ray_fov.clear();
  // visual_sub_map_cur.clear();
  // visual_converged_point.clear();
  // map_cur_frame.clear();
  // sample_points.clear();
// }

void VIOManager::computeProjectionJacobian(V3D p, MD(2, 3) & J)
{
  const double x = p[0];
  const double y = p[1];
  const double z_inv = 1. / p[2];
  const double z_inv_2 = z_inv * z_inv;
  J(0, 0) = fx * z_inv;
  J(0, 1) = 0.0;
  J(0, 2) = -fx * x * z_inv_2;
  J(1, 0) = 0.0;
  J(1, 1) = fy * z_inv;
  J(1, 2) = -fy * y * z_inv_2;
}

void VIOManager::getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level)
{
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int scale = (1 << level);
  const int u_ref_i = floorf(pc[0] / scale) * scale;
  const int v_ref_i = floorf(pc[1] / scale) * scale;
  const float subpix_u_ref = (u_ref - u_ref_i) / scale;
  const float subpix_v_ref = (v_ref - v_ref_i) / scale;
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  for (int x = 0; x < patch_size; x++)
  {
    uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i - patch_size_half * scale + x * scale) * width + (u_ref_i - patch_size_half * scale);
    for (int y = 0; y < patch_size; y++, img_ptr += scale)
    {
      patch_tmp[patch_size_total * level + x * patch_size + y] =
          w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale];
    }
  }
}

void VIOManager::insertPointIntoVoxelMap(VisualPoint *pt_new)
{
  V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
  double voxel_size = 0.5;
  float loc_xyz[3];
  for (int j = 0; j < 3; j++)
  {
    loc_xyz[j] = pt_w[j] / voxel_size;
    if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
  }
  VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
  auto iter = feat_map.find(position);
  if (iter != feat_map.end())
  {
    iter->second->voxel_points.push_back(pt_new);
    iter->second->count++;
  }
  else
  {
    VOXEL_POINTS *ot = new VOXEL_POINTS(0);
    ot->voxel_points.push_back(pt_new);
    feat_map[position] = ot;
  }
}

void VIOManager::getWarpMatrixAffineHomography(const vk::AbstractCamera &cam, const V2D &px_ref, const V3D &xyz_ref, const V3D &normal_ref,
                                                  const SE3 &T_cur_ref, const int level_ref, Matrix2d &A_cur_ref)
{
  // create homography matrix
  const V3D t = T_cur_ref.inverse().translation();
  const Eigen::Matrix3d H_cur_ref =
      T_cur_ref.rotation_matrix() * (normal_ref.dot(xyz_ref) * Eigen::Matrix3d::Identity() - t * normal_ref.transpose());
  // Compute affine warp matrix A_ref_cur using homography projection
  const int kHalfPatchSize = 4;
  V3D f_du_ref(cam.cam2world(px_ref + Eigen::Vector2d(kHalfPatchSize, 0) * (1 << level_ref)));
  V3D f_dv_ref(cam.cam2world(px_ref + Eigen::Vector2d(0, kHalfPatchSize) * (1 << level_ref)));
  //   f_du_ref = f_du_ref/f_du_ref[2];
  //   f_dv_ref = f_dv_ref/f_dv_ref[2];
  const V3D f_cur(H_cur_ref * xyz_ref);
  const V3D f_du_cur = H_cur_ref * f_du_ref;
  const V3D f_dv_cur = H_cur_ref * f_dv_ref;
  V2D px_cur(cam.world2cam(f_cur));
  V2D px_du_cur(cam.world2cam(f_du_cur));
  V2D px_dv_cur(cam.world2cam(f_dv_cur));
  A_cur_ref.col(0) = (px_du_cur - px_cur) / kHalfPatchSize;
  A_cur_ref.col(1) = (px_dv_cur - px_cur) / kHalfPatchSize;
}

void VIOManager::getWarpMatrixAffine(const vk::AbstractCamera &cam, const Vector2d &px_ref, const Vector3d &f_ref, const double depth_ref,
                                        const SE3 &T_cur_ref, const int level_ref, const int pyramid_level, const int halfpatch_size,
                                        Matrix2d &A_cur_ref)
{
  // Compute affine warp matrix A_ref_cur
  const Vector3d xyz_ref(f_ref * depth_ref);
  Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size, 0) * (1 << level_ref) * (1 << pyramid_level)));
  Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0, halfpatch_size) * (1 << level_ref) * (1 << pyramid_level)));
  xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];
  const Vector2d px_cur(cam.world2cam(T_cur_ref * (xyz_ref)));
  const Vector2d px_du(cam.world2cam(T_cur_ref * (xyz_du_ref)));
  const Vector2d px_dv(cam.world2cam(T_cur_ref * (xyz_dv_ref)));
  A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
}

void VIOManager::warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref, const int search_level,
                               const int pyramid_level, const int halfpatch_size, float *patch)
{
  const int patch_size = halfpatch_size * 2;
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  if (isnan(A_ref_cur(0, 0)))
  {
    printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
    return;
  }

  float *patch_ptr = patch;
  for (int y = 0; y < patch_size; ++y)
  {
    for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
    {
      Vector2f px_patch(x - halfpatch_size, y - halfpatch_size);
      px_patch *= (1 << search_level);
      px_patch *= (1 << pyramid_level);
      const Vector2f px(A_ref_cur * px_patch + px_ref.cast<float>());
      if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1)
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = 0;
      else
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = (float)vk::interpolateMat_8u(img_ref, px[0], px[1]);
    }
  }
}

int VIOManager::getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level)
{
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();
  while (D > 3.0 && search_level < max_level)
  {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}

double VIOManager::calculateNCC(float *ref_patch, float *cur_patch, int patch_size)
{
  double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
  double mean_ref = sum_ref / patch_size;

  double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
  double mean_curr = sum_cur / patch_size;

  double numerator = 0, demoniator1 = 0, demoniator2 = 0;
  for (int i = 0; i < patch_size; i++)
  {
    double n = (ref_patch[i] - mean_ref) * (cur_patch[i] - mean_curr);
    numerator += n;
    demoniator1 += (ref_patch[i] - mean_ref) * (ref_patch[i] - mean_ref);
    demoniator2 += (cur_patch[i] - mean_curr) * (cur_patch[i] - mean_curr);
  }
  return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

void VIOManager::retrieveFromVisualSparseMap(cv::Mat img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (feat_map.size() <= 0) return;
  double ts0 = omp_get_wtime();

  // pg_down->reserve(feat_map.size());
  // downSizeFilter.setInputCloud(pg);
  // downSizeFilter.filter(*pg_down);

  // resetRvizDisplay();
  visual_submap->reset();

  // Controls whether to include the visual submap from the previous frame.
  sub_feat_map.clear();

  float voxel_size = 0.5;

  if (!normal_en) warp_map.clear();

  cv::Mat depth_img = cv::Mat::zeros(height, width, CV_32FC1);
  float *it = (float *)depth_img.data;

  // float it[height * width] = {0.0};

  // double t_insert, t_depth, t_position;
  // t_insert=t_depth=t_position=0;

  int loc_xyz[3];

  // printf("A0. initial depthmap: %.6lf \n", omp_get_wtime() - ts0);
  // double ts1 = omp_get_wtime();

  // printf("pg size: %zu \n", pg.size());

  for (int i = 0; i < pg.size(); i++)
  {
    // double t0 = omp_get_wtime();

    V3D pt_w = pg[i].point_w;

    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = floor(pt_w[j] / voxel_size);
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

    // t_position += omp_get_wtime()-t0;
    // double t1 = omp_get_wtime();

    auto iter = sub_feat_map.find(position);
    if (iter == sub_feat_map.end()) { sub_feat_map[position] = 0; }
    else { iter->second = 0; }

    // t_insert += omp_get_wtime()-t1;
    // double t2 = omp_get_wtime();

    V3D pt_c(new_frame_->w2f(pt_w));

    if (pt_c[2] > 0)
    {
      V2D px;
      // px[0] = fx * pt_c[0]/pt_c[2] + cx;
      // px[1] = fy * pt_c[1]/pt_c[2]+ cy;
      px = new_frame_->cam_->world2cam(pt_c);

      if (new_frame_->cam_->isInFrame(px.cast<int>(), border))
      {
        // cv::circle(img_cp, cv::Point2f(px[0], px[1]), 3, cv::Scalar(0, 0, 255), -1, 8);
        float depth = pt_c[2];
        int col = int(px[0]);
        int row = int(px[1]);
        // This image is used as an occlusion z-buffer below.  Overwriting it
        // made the result depend on the order emitted by PCL's voxel filter.
        // Keep the nearest positive depth so identical point sets produce the
        // same visibility decision regardless of input ordering.
        float &pixel_depth = it[width * row + col];
        if (pixel_depth == 0.0f || depth < pixel_depth) pixel_depth = depth;
      }
    }
    // t_depth += omp_get_wtime()-t2;
  }

  // imshow("depth_img", depth_img);
  // printf("A1: %.6lf \n", omp_get_wtime() - ts1);
  // printf("A11. calculate pt position: %.6lf \n", t_position);
  // printf("A12. sub_postion.insert(position): %.6lf \n", t_insert);
  // printf("A13. generate depth map: %.6lf \n", t_depth);
  // printf("A. projection: %.6lf \n", omp_get_wtime() - ts0);

  // double t1 = omp_get_wtime();
  vector<VOXEL_LOCATION> DeleteKeyList;

  // unordered_map iteration order is an implementation detail.  Selection is
  // mostly a per-grid minimum, but a deterministic traversal also makes exact
  // ties replayable and keeps future selection changes from inheriting hash
  // bucket order.
  vector<VOXEL_LOCATION> sub_feat_keys;
  sub_feat_keys.reserve(sub_feat_map.size());
  for (const auto &entry : sub_feat_map) sub_feat_keys.push_back(entry.first);
  std::sort(sub_feat_keys.begin(), sub_feat_keys.end(), voxelLocationLess);

  for (const VOXEL_LOCATION &position : sub_feat_keys)
  {
    // double t4 = omp_get_wtime();
    auto corre_voxel = feat_map.find(position);
    // double t5 = omp_get_wtime();

    if (corre_voxel != feat_map.end())
    {
      bool voxel_in_fov = false;
      std::vector<VisualPoint *> &voxel_points = corre_voxel->second->voxel_points;
      int voxel_num = voxel_points.size();

      for (int i = 0; i < voxel_num; i++)
      {
        VisualPoint *pt = voxel_points[i];
        if (pt == nullptr) continue;
        if (pt->obs_.size() == 0) continue;

        V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt->normal_);
        V3D dir(new_frame_->T_f_w_ * pt->pos_);
        if (dir[2] < 0) continue;
        // dir.normalize();
        // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree  0.17 80 degree 0.08 85 degree

        V2D pc(new_frame_->w2c(pt->pos_));
        if (new_frame_->cam_->isInFrame(pc.cast<int>(), border))
        {
          // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 255, 255), -1, 8);
          voxel_in_fov = true;
          const int index = checkedGridIndex(pc, grid_size, grid_n_width,
                                             grid_n_height);
          if (index < 0) continue;
          grid_num[index] = TYPE_MAP;
          Vector3d obs_vec(new_frame_->pos() - pt->pos_);
          float cur_dist = obs_vec.norm();
          if (preferMapPoint(cur_dist, pt, map_dist[index],
                             retrieve_voxel_points[index]))
          {
            map_dist[index] = cur_dist;
            retrieve_voxel_points[index] = pt;
          }
        }
      }
      if (!voxel_in_fov) { DeleteKeyList.push_back(position); }
    }
  }

  // RayCasting Module
  if (raycast_en)
  {
    for (int i = 0; i < length; i++)
    {
      if (grid_num[i] == TYPE_MAP || border_flag[i] == 1) continue;

      // int row = static_cast<int>(i / grid_n_width) * grid_size + grid_size /
      // 2; int col = (i - static_cast<int>(i / grid_n_width) * grid_n_width) *
      // grid_size + grid_size / 2;

      // cv::circle(img_cp, cv::Point2f(col, row), 3, cv::Scalar(255, 255, 0),
      // -1, 8);

      // vector<V3D> sample_points_temp;
      // bool add_sample = false;

      for (const auto &it : rays_with_sample_points[i])
      {
        V3D sample_point_w = new_frame_->f2w(it);
        // sample_points_temp.push_back(sample_point_w);

        for (int j = 0; j < 3; j++)
        {
          loc_xyz[j] = floor(sample_point_w[j] / voxel_size);
          if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
        }

        VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

        auto corre_sub_feat_map = sub_feat_map.find(sample_pos);
        if (corre_sub_feat_map != sub_feat_map.end()) break;

        auto corre_feat_map = feat_map.find(sample_pos);
        if (corre_feat_map != feat_map.end())
        {
          bool voxel_in_fov = false;

          std::vector<VisualPoint *> &voxel_points = corre_feat_map->second->voxel_points;
          int voxel_num = voxel_points.size();
          if (voxel_num == 0) continue;

          for (int j = 0; j < voxel_num; j++)
          {
            VisualPoint *pt = voxel_points[j];

            if (pt == nullptr) continue;
            if (pt->obs_.size() == 0) continue;

            // sub_map_ray.push_back(pt); // cloud_visual_sub_map
            // add_sample = true;

            V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt->normal_);
            V3D dir(new_frame_->T_f_w_ * pt->pos_);
            if (dir[2] < 0) continue;
            dir.normalize();
            // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree 0.17 80 degree 0.08 85 degree

            V2D pc(new_frame_->w2c(pt->pos_));

            if (new_frame_->cam_->isInFrame(pc.cast<int>(), border))
            {
              // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(255, 255, 0), -1, 8); 
              // sub_map_ray_fov.push_back(pt);

              voxel_in_fov = true;
              const int index = checkedGridIndex(pc, grid_size, grid_n_width,
                                                 grid_n_height);
              if (index < 0) continue;
              grid_num[index] = TYPE_MAP;
              Vector3d obs_vec(new_frame_->pos() - pt->pos_);

              float cur_dist = obs_vec.norm();

              if (preferMapPoint(cur_dist, pt, map_dist[index],
                                 retrieve_voxel_points[index]))
              {
                map_dist[index] = cur_dist;
                retrieve_voxel_points[index] = pt;
              }
            }
          }

          if (voxel_in_fov) sub_feat_map[sample_pos] = 0;
          break;
        }
        else
        {
          VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
          auto iter = plane_map.find(sample_pos);
          if (iter != plane_map.end())
          {
            VoxelOctoTree *current_octo;
            current_octo = iter->second->find_correspond(sample_point_w);
            if (current_octo->plane_ptr_->is_plane_)
            {
              pointWithVar plane_center;
              VoxelPlane &plane = *current_octo->plane_ptr_;
              plane_center.point_w = plane.center_;
              plane_center.normal = plane.normal_;
              visual_submap->add_from_voxel_map.push_back(plane_center);
              break;
            }
          }
        }
      }
      // if(add_sample) sample_points.push_back(sample_points_temp);
    }
  }

  for (auto &key : DeleteKeyList)
  {
    sub_feat_map.erase(key);
  }

  // double t2 = omp_get_wtime();

  // cout<<"B. feat_map.find: "<<t2-t1<<endl;

  // double t_2, t_3, t_4, t_5;
  // t_2=t_3=t_4=t_5=0;

  for (int i = 0; i < length; i++)
  {
    if (grid_num[i] == TYPE_MAP)
    {
      // double t_1 = omp_get_wtime();

      VisualPoint *pt = retrieve_voxel_points[i];
      // visual_sub_map_cur.push_back(pt); // before

      V2D pc(new_frame_->w2c(pt->pos_));

      // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 0, 255), -1, 8); // Green Sparse Align tracked

      V3D pt_cam(new_frame_->w2f(pt->pos_));
      bool depth_continous = false;
      for (int u = -patch_size_half; u <= patch_size_half; u++)
      {
        for (int v = -patch_size_half; v <= patch_size_half; v++)
        {
          if (u == 0 && v == 0) continue;

          float depth = it[width * (v + int(pc[1])) + u + int(pc[0])];

          if (depth == 0.) continue;

          double delta_dist = abs(pt_cam[2] - depth);

          if (delta_dist > 0.5)
          {
            depth_continous = true;
            break;
          }
        }
        if (depth_continous) break;
      }
      if (depth_continous) continue;

      // t_2 += omp_get_wtime() - t_1;

      // t_1 = omp_get_wtime();
      Feature *ref_ftr;
      std::vector<float> patch_wrap(warp_len);

      int search_level;
      Matrix2d A_cur_ref_zero;

      if (!pt->is_normal_initialized_) continue;

      if (normal_en)
      {
        float phtometric_errors_min = std::numeric_limits<float>::max();

        if (pt->obs_.size() == 1)
        {
          ref_ftr = *pt->obs_.begin();
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        }
        else if (!pt->has_ref_patch_)
        {
          for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it)
          {
            Feature *ref_patch_temp = *it;
            float *patch_temp = ref_patch_temp->patch_;
            float phtometric_errors = 0.0;
            int count = 0;
            for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm)
            {
              if ((*itm)->id_ == ref_patch_temp->id_) continue;
              float *patch_cache = (*itm)->patch_;

              for (int ind = 0; ind < patch_size_total; ind++)
              {
                phtometric_errors += (patch_temp[ind] - patch_cache[ind]) * (patch_temp[ind] - patch_cache[ind]);
              }
              count++;
            }
            phtometric_errors = phtometric_errors / count;
            if (phtometric_errors < phtometric_errors_min)
            {
              phtometric_errors_min = phtometric_errors;
              ref_ftr = ref_patch_temp;
            }
          }
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        }
        else { ref_ftr = pt->ref_patch; }
      }
      else
      {
        if (!pt->getCloseViewObs(new_frame_->pos(), ref_ftr, pc)) continue;
      }

      if (normal_en)
      {
        V3D norm_vec = (ref_ftr->T_f_w_.rotation_matrix() * pt->normal_).normalized();
        
        V3D pf(ref_ftr->T_f_w_ * pt->pos_);
        // V3D pf_norm = pf.normalized();
        
        // double cos_theta = norm_vec.dot(pf_norm);
        // if(cos_theta < 0) norm_vec = -norm_vec;
        // if (abs(cos_theta) < 0.08) continue; // 0.5 60 degree 0.34 70 degree 0.17 80 degree 0.08 85 degree

        SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();

        getWarpMatrixAffineHomography(*cam, ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref_zero);

        search_level = getBestSearchLevel(A_cur_ref_zero, 2);
      }
      else
      {
        auto iter_warp = warp_map.find(ref_ftr->id_);
        if (iter_warp != warp_map.end())
        {
          search_level = iter_warp->second->search_level;
          A_cur_ref_zero = iter_warp->second->A_cur_ref;
        }
        else
        {
          getWarpMatrixAffine(*cam, ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(), new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(),
                              ref_ftr->level_, 0, patch_size_half, A_cur_ref_zero);

          search_level = getBestSearchLevel(A_cur_ref_zero, 2);

          Warp *ot = new Warp(search_level, A_cur_ref_zero);
          warp_map[ref_ftr->id_] = ot;
        }
      }
      // t_4 += omp_get_wtime() - t_1;

      // t_1 = omp_get_wtime();

      for (int pyramid_level = 0; pyramid_level <= patch_pyrimid_level - 1; pyramid_level++)
      {
        warpAffine(A_cur_ref_zero, ref_ftr->img_, ref_ftr->px_, ref_ftr->level_, search_level, pyramid_level, patch_size_half, patch_wrap.data());
      }

      getImagePatch(img, pc, patch_buffer.data(), 0);

      float error = 0.0;
      for (int ind = 0; ind < patch_size_total; ind++)
      {
        error += (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time * patch_buffer[ind]) *
                 (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time * patch_buffer[ind]);
      }

      if (ncc_en)
      {
        double ncc = calculateNCC(patch_wrap.data(), patch_buffer.data(), patch_size_total);
        if (ncc < ncc_thre)
        {
          // grid_num[i] = TYPE_UNKNOWN;
          continue;
        }
      }

      if (error > outlier_threshold * patch_size_total) continue;

      visual_submap->voxel_points.push_back(pt);
      visual_submap->propa_errors.push_back(error);
      visual_submap->search_levels.push_back(search_level);
      visual_submap->errors.push_back(error);
      visual_submap->warp_patch.push_back(patch_wrap);
      visual_submap->inv_expo_list.push_back(ref_ftr->inv_expo_time_);

      // t_5 += omp_get_wtime() - t_1;
    }
  }
  total_points = visual_submap->voxel_points.size();

  // double t3 = omp_get_wtime();
  // cout<<"C. addSubSparseMap: "<<t3-t2<<endl;
  // cout<<"depthcontinuous: C1 "<<t_2<<" C2 "<<t_3<<" C3 "<<t_4<<" C4
  // "<<t_5<<endl;
  if (verbose) printf("[ VIO ] Retrieve %d points from visual sparse map\n", total_points);
}

void VIOManager::computeJacobianAndUpdateEKF(cv::Mat img)
{
  if (total_points == 0) return;

  // Do not let a frame with no valid photometric measurements reuse the gain
  // from a previous frame when the caller applies the covariance update.
  G.setZero();
  
  compute_jacobian_time = update_ekf_time = 0.0;

  for (int level = patch_pyrimid_level - 1; level >= 0; level--)
  {
    if (inverse_composition_en)
    {
      has_ref_patch_cache = false;
      updateStateInverse(img, level);
    }
    else
      updateState(img, level);
  }
  state->cov -= G * state->cov;
  updateFrameState(*state);
}

void VIOManager::generateVisualMapPoints(cv::Mat img, vector<pointWithVar> &pg)
{
  if (pg.size() <= 10) return;

  // double t0 = omp_get_wtime();
  for (int i = 0; i < pg.size(); i++)
  {
    if (pg[i].normal == V3D(0, 0, 0)) continue;

    V3D pt = pg[i].point_w;
    V2D pc(new_frame_->w2c(pt));

    if (new_frame_->cam_->isInFrame(pc.cast<int>(), border)) // 20px is the patch size in the matcher
    {
      const int index = checkedGridIndex(pc, grid_size, grid_n_width,
                                         grid_n_height);
      if (index < 0) continue;

      if (grid_num[index] != TYPE_MAP)
      {
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        // if (cur_value < 5) continue;
        if (preferPointCandidate(cur_value, pg[i], grid_num[index],
                                 scan_value[index], append_voxel_points[index]))
        {
          scan_value[index] = cur_value;
          append_voxel_points[index] = pg[i];
          grid_num[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  for (int j = 0; j < visual_submap->add_from_voxel_map.size(); j++)
  {
    V3D pt = visual_submap->add_from_voxel_map[j].point_w;
    V2D pc(new_frame_->w2c(pt));

    if (new_frame_->cam_->isInFrame(pc.cast<int>(), border)) // 20px is the patch size in the matcher
    {
      const int index = checkedGridIndex(pc, grid_size, grid_n_width,
                                         grid_n_height);
      if (index < 0) continue;

      if (grid_num[index] != TYPE_MAP)
      {
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        if (preferPointCandidate(cur_value,
                                 visual_submap->add_from_voxel_map[j],
                                 grid_num[index], scan_value[index],
                                 append_voxel_points[index]))
        {
          scan_value[index] = cur_value;
          append_voxel_points[index] = visual_submap->add_from_voxel_map[j];
          grid_num[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  // double t_b1 = omp_get_wtime() - t0;
  // t0 = omp_get_wtime();

  int add = 0;
  for (int i = 0; i < length; i++)
  {
    if (grid_num[i] == TYPE_POINTCLOUD) // && (scan_value[i]>=50))
    {
      pointWithVar pt_var = append_voxel_points[i];
      V3D pt = pt_var.point_w;

      V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt_var.normal);
      V3D dir(new_frame_->T_f_w_ * pt);
      dir.normalize();
      double cos_theta = dir.dot(norm_vec);
      // if(std::fabs(cos_theta)<0.34) continue; // 70 degree
      V2D pc(new_frame_->w2c(pt));

      float *patch = new float[patch_size_total];
      getImagePatch(img, pc, patch, 0);

      VisualPoint *pt_new = new VisualPoint(pt);

      Vector3d f = cam->cam2world(pc);
      Feature *ftr_new = new Feature(pt_new, patch, pc, f, new_frame_->T_f_w_, 0);
      ftr_new->img_ = img;
      ftr_new->id_ = new_frame_->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time;

      pt_new->addFrameRef(ftr_new);
      pt_new->covariance_ = pt_var.var;
      pt_new->is_normal_initialized_ = true;

      if (cos_theta < 0) { pt_new->normal_ = -pt_var.normal; }
      else { pt_new->normal_ = pt_var.normal; }
      
      pt_new->previous_normal_ = pt_new->normal_;

      insertPointIntoVoxelMap(pt_new);
      add += 1;
      // map_cur_frame.push_back(pt_new);
    }
  }

  // double t_b2 = omp_get_wtime() - t0;

  if (verbose) printf("[ VIO ] Append %d new visual map points\n", add);
  // printf("pg.size: %d \n", pg.size());
  // printf("B1. : %.6lf \n", t_b1);
  // printf("B2. : %.6lf \n", t_b2);
}

void VIOManager::updateVisualMapPoints(cv::Mat img)
{
  if (total_points == 0) return;

  int update_num = 0;
  SE3 pose_cur = new_frame_->T_f_w_;
  for (int i = 0; i < total_points; i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];
    if (pt == nullptr) continue;
    if (pt->is_converged_)
    { 
      pt->deleteNonRefPatchFeatures();
      continue;
    }

    V2D pc(new_frame_->w2c(pt->pos_));
    if (!patchSamplingInBounds(pc, 1, patch_size, patch_size_half, img,
                               width, height, false))
      continue;
    bool add_flag = false;
    
    // TODO: condition: distance and view_angle
    // Step 1: time
    Feature *last_feature = pt->obs_.back();
    // if(new_frame_->id_ >= last_feature->id_ + 10) add_flag = true; // 10

    // Step 2: delta_pose
    SE3 pose_ref = last_feature->T_f_w_;
    SE3 delta_pose = pose_ref * pose_cur.inverse();
    double delta_p = delta_pose.translation().norm();
    double delta_theta = (delta_pose.rotation_matrix().trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (delta_pose.rotation_matrix().trace() - 1));
    if (delta_p > 0.5 || delta_theta > 0.3) add_flag = true; // 0.5 || 0.3

    // Step 3: pixel distance
    Vector2d last_px = last_feature->px_;
    double pixel_dist = (pc - last_px).norm();
    if (pixel_dist > 40) add_flag = true;

    // Maintain the size of 3D point observation features.
    if (pt->obs_.size() >= 30)
    {
      Feature *ref_ftr;
      pt->findMinScoreFeature(new_frame_->pos(), ref_ftr);
      pt->deleteFeatureRef(ref_ftr);
      // cout<<"pt->obs_.size() exceed 20 !!!!!!"<<endl;
    }
    if (add_flag)
    {
      update_num += 1;
      update_flag[i] = 1;
      float *patch_temp = new float[patch_size_total];
      getImagePatch(img, pc, patch_temp, 0);
      Vector3d f = cam->cam2world(pc);
      Feature *ftr_new = new Feature(pt, patch_temp, pc, f, new_frame_->T_f_w_, visual_submap->search_levels[i]);
      ftr_new->img_ = img;
      ftr_new->id_ = new_frame_->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time;
      pt->addFrameRef(ftr_new);
    }
  }
  if (verbose) printf("[ VIO ] Update %d points in visual submap\n", update_num);
}

void VIOManager::updateReferencePatch(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (total_points == 0) return;

  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (!pt->is_normal_initialized_) continue;
    if (pt->is_converged_) continue;
    if (pt->obs_.size() <= 5) continue;
    if (update_flag[i] == 0) continue;

    const V3D &p_w = pt->pos_;
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_w[j] / 0.5;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = plane_map.find(position);
    if (iter != plane_map.end())
    {
      VoxelOctoTree *current_octo;
      current_octo = iter->second->find_correspond(p_w);
      if (current_octo->plane_ptr_->is_plane_)
      {
        VoxelPlane &plane = *current_octo->plane_ptr_;
        float dis_to_plane = plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_;
        float dis_to_plane_abs = fabs(dis_to_plane);
        float dis_to_center = (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) +
                              (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) + (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));
        float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);
        if (range_dis <= 3 * plane.radius_)
        {
          Eigen::Matrix<double, 1, 6> J_nq;
          J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
          J_nq.block<1, 3>(0, 3) = -plane.normal_;
          double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
          sigma_l += plane.normal_.transpose() * pt->covariance_ * plane.normal_;

          if (dis_to_plane_abs < 3 * sqrt(sigma_l))
          {
            // V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * plane.normal_);
            // V3D pf(new_frame_->T_f_w_ * pt->pos_);
            // V3D pf_ref(pt->ref_patch->T_f_w_ * pt->pos_);
            // V3D norm_vec_ref(pt->ref_patch->T_f_w_.rotation_matrix() *
            // plane.normal); double cos_ref = pf_ref.dot(norm_vec_ref);
            
            if (pt->previous_normal_.dot(plane.normal_) < 0) { pt->normal_ = -plane.normal_; }
            else { pt->normal_ = plane.normal_; }

            double normal_update = (pt->normal_ - pt->previous_normal_).norm();

            pt->previous_normal_ = pt->normal_;

            if (normal_update < 0.0001 && pt->obs_.size() > 10)
            {
              pt->is_converged_ = true;
              // visual_converged_point.push_back(pt);
            }
          }
        }
      }
    }

    float score_max = -1000.;
    for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it)
    {
      Feature *ref_patch_temp = *it;
      float *patch_temp = ref_patch_temp->patch_;
      float NCC_up = 0.0;
      float NCC_down1 = 0.0;
      float NCC_down2 = 0.0;
      float NCC = 0.0;
      float score = 0.0;
      int count = 0;

      V3D pf = ref_patch_temp->T_f_w_ * pt->pos_;
      V3D norm_vec = ref_patch_temp->T_f_w_.rotation_matrix() * pt->normal_;
      pf.normalize();
      double cos_angle = pf.dot(norm_vec);
      // if(fabs(cos_angle) < 0.86) continue; // 20 degree

      float ref_mean;
      if (abs(ref_patch_temp->mean_) < 1e-6)
      {
        float ref_sum = std::accumulate(patch_temp, patch_temp + patch_size_total, 0.0);
        ref_mean = ref_sum / patch_size_total;
        ref_patch_temp->mean_ = ref_mean;
      }

      for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm)
      {
        if ((*itm)->id_ == ref_patch_temp->id_) continue;
        float *patch_cache = (*itm)->patch_;

        float other_mean;
        if (abs((*itm)->mean_) < 1e-6)
        {
          float other_sum = std::accumulate(patch_cache, patch_cache + patch_size_total, 0.0);
          other_mean = other_sum / patch_size_total;
          (*itm)->mean_ = other_mean;
        }

        for (int ind = 0; ind < patch_size_total; ind++)
        {
          NCC_up += (patch_temp[ind] - ref_mean) * (patch_cache[ind] - other_mean);
          NCC_down1 += (patch_temp[ind] - ref_mean) * (patch_temp[ind] - ref_mean);
          NCC_down2 += (patch_cache[ind] - other_mean) * (patch_cache[ind] - other_mean);
        }
        NCC += fabs(NCC_up / sqrt(NCC_down1 * NCC_down2));
        count++;
      }

      NCC = NCC / count;

      score = NCC + cos_angle;

      ref_patch_temp->score_ = score;

      if (score > score_max)
      {
        score_max = score;
        pt->ref_patch = ref_patch_temp;
        pt->has_ref_patch_ = true;
      }
    }

  }
}

void VIOManager::projectPatchFromRefToCur(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (total_points == 0) return;
  // if(new_frame_->id_ != 2) return; //124

  int patch_size = 25;
  string dir = string(ROOT_DIR) + "Log/ref_cur_combine/";

  cv::Mat result = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat result_normal = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat result_dense = cv::Mat::zeros(height, width, CV_8UC1);

  cv::Mat img_photometric_error = new_frame_->img_.clone();

  uchar *it = (uchar *)result.data;
  uchar *it_normal = (uchar *)result_normal.data;
  uchar *it_dense = (uchar *)result_dense.data;

  struct pixel_member
  {
    Vector2f pixel_pos;
    uint8_t pixel_value;
  };

  int num = 0;
  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (pt->is_normal_initialized_)
    {
      Feature *ref_ftr;
      ref_ftr = pt->ref_patch;
      // Feature* ref_ftr;
      V2D pc(new_frame_->w2c(pt->pos_));
      V2D pc_prior(new_frame_->w2c_prior(pt->pos_));

      V3D norm_vec(ref_ftr->T_f_w_.rotation_matrix() * pt->normal_);
      V3D pf(ref_ftr->T_f_w_ * pt->pos_);

      if (pf.dot(norm_vec) < 0) norm_vec = -norm_vec;

      // norm_vec << norm_vec(1), norm_vec(0), norm_vec(2);
      cv::Mat img_cur = new_frame_->img_;
      cv::Mat img_ref = ref_ftr->img_;

      SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();
      Matrix2d A_cur_ref;
      getWarpMatrixAffineHomography(*cam, ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref);

      // const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
      int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);

      double D = A_cur_ref.determinant();
      if (D > 3) continue;

      num++;

      cv::Mat ref_cur_combine_temp;
      int radius = 20;
      cv::hconcat(img_cur, img_ref, ref_cur_combine_temp);
      cv::cvtColor(ref_cur_combine_temp, ref_cur_combine_temp, CV_GRAY2BGR);

      getImagePatch(img_cur, pc, patch_buffer.data(), 0);

      float error_est = 0.0;
      float error_gt = 0.0;

      for (int ind = 0; ind < patch_size_total; ind++)
      {
        error_est += (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time * patch_buffer[ind]) *
                     (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time * patch_buffer[ind]);
      }
      std::string ref_est = "ref_est " + std::to_string(1.0 / ref_ftr->inv_expo_time_);
      std::string cur_est = "cur_est " + std::to_string(1.0 / state->inv_expo_time);
      std::string cur_propa = "cur_gt " + std::to_string(error_gt);
      std::string cur_optimize = "cur_est " + std::to_string(error_est);

      cv::putText(ref_cur_combine_temp, ref_est, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - 40, ref_ftr->px_[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4,
                  cv::Scalar(0, 255, 0), 1, 8, 0);

      cv::putText(ref_cur_combine_temp, cur_est, cv::Point2f(pc[0] - 40, pc[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8, 0);
      cv::putText(ref_cur_combine_temp, cur_propa, cv::Point2f(pc[0] - 40, pc[1] + 60), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, 255), 1, 8,
                  0);
      cv::putText(ref_cur_combine_temp, cur_optimize, cv::Point2f(pc[0] - 40, pc[1] + 80), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8,
                  0);

      cv::rectangle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - radius, ref_ftr->px_[1] - radius),
                    cv::Point2f(ref_ftr->px_[0] + img_cur.cols + radius, ref_ftr->px_[1] + radius), cv::Scalar(0, 0, 255), 1);
      cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc[0] - radius, pc[1] - radius), cv::Point2f(pc[0] + radius, pc[1] + radius),
                    cv::Scalar(0, 255, 0), 1);
      cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc_prior[0] - radius, pc_prior[1] - radius),
                    cv::Point2f(pc_prior[0] + radius, pc_prior[1] + radius), cv::Scalar(255, 255, 255), 1);
      cv::circle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols, ref_ftr->px_[1]), 1, cv::Scalar(0, 0, 255), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc[0], pc[1]), 1, cv::Scalar(0, 255, 0), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc_prior[0], pc_prior[1]), 1, cv::Scalar(255, 255, 255), -1, 8);
      cv::imwrite(dir + std::to_string(new_frame_->id_) + "_" + std::to_string(ref_ftr->id_) + "_" + std::to_string(num) + ".png",
                  ref_cur_combine_temp);

      std::vector<std::vector<pixel_member>> pixel_warp_matrix;

      for (int y = 0; y < patch_size; ++y)
      {
        vector<pixel_member> pixel_warp_vec;
        for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
        {
          Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
          px_patch *= (1 << search_level);
          const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
          uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

          const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
          if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
            continue;
          else
          {
            pixel_member pixel_warp;
            pixel_warp.pixel_pos << px[0], px[1];
            pixel_warp.pixel_value = pixel_value;
            pixel_warp_vec.push_back(pixel_warp);
          }
        }
        pixel_warp_matrix.push_back(pixel_warp_vec);
      }

      float x_min = 1000;
      float y_min = 1000;
      float x_max = 0;
      float y_max = 0;

      for (int i = 0; i < pixel_warp_matrix.size(); i++)
      {
        vector<pixel_member> pixel_warp_row = pixel_warp_matrix[i];
        for (int j = 0; j < pixel_warp_row.size(); j++)
        {
          float x_temp = pixel_warp_row[j].pixel_pos[0];
          float y_temp = pixel_warp_row[j].pixel_pos[1];
          if (x_temp < x_min) x_min = x_temp;
          if (y_temp < y_min) y_min = y_temp;
          if (x_temp > x_max) x_max = x_temp;
          if (y_temp > y_max) y_max = y_temp;
        }
      }
      int x_min_i = floor(x_min);
      int y_min_i = floor(y_min);
      int x_max_i = ceil(x_max);
      int y_max_i = ceil(y_max);
      Matrix2f A_cur_ref_Inv = A_cur_ref.inverse().cast<float>();
      for (int i = x_min_i; i < x_max_i; i++)
      {
        for (int j = y_min_i; j < y_max_i; j++)
        {
          Eigen::Vector2f pc_temp(i, j);
          Vector2f px_patch = A_cur_ref_Inv * (pc_temp - pc.cast<float>());
          if (px_patch[0] > (-patch_size / 2 * (1 << search_level)) && px_patch[0] < (patch_size / 2 * (1 << search_level)) &&
              px_patch[1] > (-patch_size / 2 * (1 << search_level)) && px_patch[1] < (patch_size / 2 * (1 << search_level)))
          {
            const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
            uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);
            it_normal[width * j + i] = pixel_value;
          }
        }
      }
    }
  }
  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (!pt->is_normal_initialized_) continue;

    Feature *ref_ftr;
    V2D pc(new_frame_->w2c(pt->pos_));
    ref_ftr = pt->ref_patch;

    Matrix2d A_cur_ref;
    getWarpMatrixAffine(*cam, ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(), new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(), 0, 0,
                        patch_size_half, A_cur_ref);
    int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);
    double D = A_cur_ref.determinant();
    if (D > 3) continue;

    cv::Mat img_cur = new_frame_->img_;
    cv::Mat img_ref = ref_ftr->img_;
    for (int y = 0; y < patch_size; ++y)
    {
      for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
      {
        Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
        px_patch *= (1 << search_level);
        const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
        uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

        const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
        if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
          continue;
        else
        {
          int col = int(px[0]);
          int row = int(px[1]);
          it[width * row + col] = pixel_value;
        }
      }
    }
  }
  cv::Mat ref_cur_combine;
  cv::Mat ref_cur_combine_normal;
  cv::Mat ref_cur_combine_error;

  cv::hconcat(result, new_frame_->img_, ref_cur_combine);
  cv::hconcat(result_normal, new_frame_->img_, ref_cur_combine_normal);

  cv::cvtColor(ref_cur_combine, ref_cur_combine, CV_GRAY2BGR);
  cv::cvtColor(ref_cur_combine_normal, ref_cur_combine_normal, CV_GRAY2BGR);
  cv::absdiff(img_photometric_error, result_normal, img_photometric_error);
  cv::hconcat(img_photometric_error, new_frame_->img_, ref_cur_combine_error);

  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + ".png", ref_cur_combine);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + +"_0_" +
                  "photometric"
                  ".png",
              ref_cur_combine_error);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + "normal" + ".png", ref_cur_combine_normal);
}

void VIOManager::precomputeReferencePatches(int level)
{
  double t1 = omp_get_wtime();
  if (total_points == 0) return;
  MD(1, 2) Jimg;
  MD(2, 3) Jdpi;
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;

  const int H_DIM = total_points * patch_size_total;

  H_sub_inv.resize(H_DIM, 6);
  H_sub_inv.setZero();
  M3D p_w_hat;

  if (level < 0 || level >= 30)
  {
    has_ref_patch_cache = true;
    return;
  }
  const int scale = (1 << level);

  for (int i = 0; i < total_points; i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];
    if (pt == nullptr || pt->ref_patch == nullptr || !pt->pos_.allFinite())
      continue;
    Feature *ref_patch = pt->ref_patch;
    cv::Mat img = ref_patch->img_;

    const double depth = (pt->pos_ - ref_patch->pos()).norm();
    V3D pf = ref_patch->f_ * depth;
    V2D pc = ref_patch->px_;
    if (!std::isfinite(depth) || !hasFinitePositiveDepth(pf) ||
        !patchSamplingInBounds(pc, scale, patch_size, patch_size_half, img,
                               width, height, true))
      continue;
    M3D R_ref_w = ref_patch->T_f_w_.rotation_matrix();

    computeProjectionJacobian(pf, Jdpi);
    p_w_hat << SKEW_SYM_MATRX(pt->pos_);

    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(pc[0] / scale) * scale;
    const int v_ref_i = floorf(pc[1] / scale) * scale;
    const float subpix_u_ref = (u_ref - u_ref_i) / scale;
    const float subpix_v_ref = (v_ref - v_ref_i) / scale;
    const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
    const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;

    for (int x = 0; x < patch_size; x++)
    {
      uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
      for (int y = 0; y < patch_size; ++y, img_ptr += scale)
      {
        float du =
            0.5f *
            ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] + w_ref_bl * img_ptr[scale * width + scale] +
              w_ref_br * img_ptr[scale * width + scale * 2]) -
             (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] + w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
        float dv =
            0.5f *
            ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] + w_ref_bl * img_ptr[width * scale * 2] +
              w_ref_br * img_ptr[width * scale * 2 + scale]) -
             (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

        Jimg << du, dv;
        Jimg = Jimg * (1.0 / scale);

        JdR = Jimg * Jdpi * R_ref_w * p_w_hat;
        Jdt = -Jimg * Jdpi * R_ref_w;

        H_sub_inv.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
      }
    }
  }
  has_ref_patch_cache = true;
}

void VIOManager::updateStateInverse(cv::Mat img, int level)
{
  if (total_points == 0 || level < 0 || level >= 30) return;
  StatesGroup old_state = (*state);
  V2D pc;
  MD(1, 2) Jimg;
  MD(2, 3) Jdpi;
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;
  VectorXd z;
  MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();
  compute_jacobian_time = update_ekf_time = 0.0;
  M3D P_wi_hat;
  bool z_init = true;
  const int H_DIM = total_points * patch_size_total;

  z.resize(H_DIM);
  z.setZero();

  H_sub.resize(H_DIM, 6);
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double t1 = omp_get_wtime();
    double count_outlier = 0;
    if (has_ref_patch_cache == false) precomputeReferencePatches(level);
    int n_meas = 0;
    float error = 0.0;
    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    P_wi_hat << SKEW_SYM_MATRX(Pwi);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;

    z.setZero();
    H_sub.setZero();

    M3D p_hat;

    for (int i = 0; i < total_points; i++)
    {
      float patch_error = 0.0;

      const int scale = (1 << level);

      if (i >= static_cast<int>(visual_submap->warp_patch.size()) ||
          i >= static_cast<int>(visual_submap->errors.size()))
        continue;

      VisualPoint *pt = visual_submap->voxel_points[i];

      if (pt == nullptr || pt->ref_patch == nullptr || !pt->pos_.allFinite())
        continue;

      Feature *ref_patch = pt->ref_patch;
      const double ref_depth = (pt->pos_ - ref_patch->pos()).norm();
      const V3D ref_pf = ref_patch->f_ * ref_depth;
      if (!std::isfinite(ref_depth) || !hasFinitePositiveDepth(ref_pf) ||
          !patchSamplingInBounds(ref_patch->px_, scale, patch_size,
                                 patch_size_half, ref_patch->img_, width,
                                 height, true))
        continue;

      V3D pf = Rcw * pt->pos_ + Pcw;
      if (!hasFinitePositiveDepth(pf)) continue;
      pc = cam->world2cam(pf);
      if (!patchSamplingInBounds(pc, scale, patch_size, patch_size_half, img,
                                 width, height, false))
        continue;

      const float u_ref = pc[0];
      const float v_ref = pc[1];
      const int u_ref_i = floorf(pc[0] / scale) * scale;
      const int v_ref_i = floorf(pc[1] / scale) * scale;
      const float subpix_u_ref = (u_ref - u_ref_i) / scale;
      const float subpix_v_ref = (v_ref - v_ref_i) / scale;
      const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      const float w_ref_br = subpix_u_ref * subpix_v_ref;

      vector<float> P = visual_submap->warp_patch[i];
      if (P.size() < static_cast<size_t>(patch_size_total * (level + 1)))
        continue;
      for (int x = 0; x < patch_size; x++)
      {
        uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale)
        {
          double res = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] +
                       w_ref_br * img_ptr[scale * width + scale] - P[patch_size_total * level + x * patch_size + y];
          z(i * patch_size_total + x * patch_size + y) = res;
          patch_error += res * res;
          MD(1, 3) J_dR = H_sub_inv.block<1, 3>(i * patch_size_total + x * patch_size + y, 0);
          MD(1, 3) J_dt = H_sub_inv.block<1, 3>(i * patch_size_total + x * patch_size + y, 3);
          JdR = J_dR * Rwi + J_dt * P_wi_hat * Rwi;
          Jdt = J_dt * Rwi;
          H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
          n_meas++;
        }
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
    }

    if (n_meas == 0)
    {
      (*state) = old_state;
      break;
    }
    error = error / n_meas;

    compute_jacobian_time += omp_get_wtime() - t1;

    double t3 = omp_get_wtime();

    if (error <= last_error)
    {
      old_state = (*state);
      last_error = error;

      auto &&H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<6, 6>(0, 0) = H_sub_T * H_sub;
      MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      auto &&HTz = H_sub_T * z;
      auto vec = (*state_propagat) - (*state);
      G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
      auto solution = -K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec - G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);
      (*state) += solution;
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);

      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f)) { EKF_end = true; }
    }
    else
    {
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    if (iteration == max_iterations || EKF_end) break; 
  }
}

void VIOManager::updateState(cv::Mat img, int level)
{
  if (total_points == 0 || level < 0 || level >= 30) return;
  StatesGroup old_state = (*state);

  VectorXd z;
  MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();

  const int H_DIM = total_points * patch_size_total;
  z.resize(H_DIM);
  z.setZero();
  H_sub.resize(H_DIM, 7);
  H_sub.setZero();
  std::vector<uint8_t> valid_measurement(total_points, 0);

  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double t1 = omp_get_wtime();

    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
    Jdp_dt = Rci * Rwi.transpose();
    z.setZero();
    H_sub.setZero();
    std::fill(valid_measurement.begin(), valid_measurement.end(), 0);
    
    // int max_threads = omp_get_max_threads();
    // int desired_threads = std::min(max_threads, total_points);
    // omp_set_num_threads(desired_threads);
  
    #ifdef MP_EN
      omp_set_num_threads(MP_PROC_NUM);
      #pragma omp parallel for
    #endif
    for (int i = 0; i < total_points; i++)
    {
      // printf("thread is %d, i=%d, i address is %p\n", omp_get_thread_num(), i, &i);
      MD(1, 2) Jimg;
      MD(2, 3) Jdpi;
      MD(1, 3) Jdphi, Jdp, JdR, Jdt;

      float patch_error = 0.0;
      if (i >= static_cast<int>(visual_submap->search_levels.size()) ||
          i >= static_cast<int>(visual_submap->warp_patch.size()) ||
          i >= static_cast<int>(visual_submap->inv_expo_list.size()) ||
          i >= static_cast<int>(visual_submap->errors.size()))
        continue;
      int search_level = visual_submap->search_levels[i];
      int pyramid_level = level + search_level;
      if (pyramid_level < 0 || pyramid_level >= 30) continue;
      int scale = (1 << pyramid_level);
      float inv_scale = 1.0f / scale;

      VisualPoint *pt = visual_submap->voxel_points[i];

      if (pt == nullptr || !pt->pos_.allFinite()) continue;

      V3D pf = Rcw * pt->pos_ + Pcw;
      if (!hasFinitePositiveDepth(pf)) continue;
      V2D pc = cam->world2cam(pf);
      if (!patchSamplingInBounds(pc, scale, patch_size, patch_size_half, img,
                                 width, height, true))
        continue;

      computeProjectionJacobian(pf, Jdpi);
      M3D p_hat;
      p_hat << SKEW_SYM_MATRX(pf);

      float u_ref = pc[0];
      float v_ref = pc[1];
      int u_ref_i = floorf(pc[0] / scale) * scale;
      int v_ref_i = floorf(pc[1] / scale) * scale;
      float subpix_u_ref = (u_ref - u_ref_i) / scale;
      float subpix_v_ref = (v_ref - v_ref_i) / scale;
      float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      float w_ref_br = subpix_u_ref * subpix_v_ref;

      vector<float> P = visual_submap->warp_patch[i];
      if (level < 0 ||
          P.size() < static_cast<size_t>(patch_size_total * (level + 1)))
        continue;
      double inv_ref_expo = visual_submap->inv_expo_list[i];
      // ROS_ERROR("inv_ref_expo: %.3lf, state->inv_expo_time: %.3lf\n", inv_ref_expo, state->inv_expo_time);

      for (int x = 0; x < patch_size; x++)
      {
        uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale)
        {
          float du =
              0.5f *
              ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] + w_ref_bl * img_ptr[scale * width + scale] +
                w_ref_br * img_ptr[scale * width + scale * 2]) -
               (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] + w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
          float dv =
              0.5f *
              ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] + w_ref_bl * img_ptr[width * scale * 2] +
                w_ref_br * img_ptr[width * scale * 2 + scale]) -
               (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

          Jimg << du, dv;
          Jimg = Jimg * state->inv_expo_time;
          Jimg = Jimg * inv_scale;
          Jdphi = Jimg * Jdpi * p_hat;
          Jdp = -Jimg * Jdpi;
          JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
          Jdt = Jdp * Jdp_dt;

          double cur_value =
              w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale];
          double res = state->inv_expo_time * cur_value - inv_ref_expo * P[patch_size_total * level + x * patch_size + y];

          z(i * patch_size_total + x * patch_size + y) = res;

          patch_error += res * res;
          if (exposure_estimate_en) { H_sub.block<1, 7>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt, cur_value; }
          else { H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt; }
        }
      }
      visual_submap->errors[i] = patch_error;
      valid_measurement[i] = 1;
    }

    // Keep the expensive Jacobian/residual evaluation parallel, but make the
    // scalar objective deterministic.  An OpenMP float reduction changes its
    // summation order between runs; this value controls the accept/rollback
    // branch below, so a sub-ULP difference can send the iterative EKF down a
    // different trajectory.  Summing one patch error per point in index order
    // is cheap and gives replayable update decisions.
    double error_sum = 0.0;
    int n_meas = 0;
    for (int i = 0; i < total_points; ++i)
    {
      if (!valid_measurement[i]) continue;
      error_sum += static_cast<double>(visual_submap->errors[i]);
      n_meas += patch_size_total;
    }
    if (n_meas == 0)
    {
      (*state) = old_state;
      break;
    }
    const float error = n_meas > 0
                            ? static_cast<float>(error_sum / static_cast<double>(n_meas))
                            : std::numeric_limits<float>::infinity();
    
    compute_jacobian_time += omp_get_wtime() - t1;

    // printf("\nPYRAMID LEVEL %i\n---------------\n", level);
    // std::cout << "It. " << iteration
    //           << "\t last_error = " << last_error
    //           << "\t new_error = " << error
    //           << std::endl;

    double t3 = omp_get_wtime();

    if (error <= last_error)
    {
      old_state = (*state);
      last_error = error;

      // K = (H.transpose() / img_point_cov * H + state->cov.inverse()).inverse() * H.transpose() / img_point_cov; auto
      // vec = (*state_propagat) - (*state); G = K*H;
      // (*state) += (-K*z + vec - G*vec);

      auto &&H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<7, 7>(0, 0) = H_sub_T * H_sub;
      const MD(6, 6) pose_info = 0.5 * (H_T_H.block<6, 6>(0, 0) +
                                             H_T_H.block<6, 6>(0, 0).transpose());
      Eigen::SelfAdjointEigenSolver<M3D> trans_solver(pose_info.block<3, 3>(3, 3));
      Eigen::SelfAdjointEigenSolver<M3D> rot_solver(pose_info.block<3, 3>(0, 0));
      Eigen::SelfAdjointEigenSolver<MD(6, 6)> full_solver(pose_info);
      const auto trans_eigs = trans_solver.eigenvalues();
      const auto rot_eigs = rot_solver.eigenvalues();
      const auto full_eigs = full_solver.eigenvalues();
      last_translation_info_ratio = std::max(0.0, trans_eigs[0]) /
                                    std::max(1e-12, trans_eigs[2]);
      last_rotation_info_ratio = std::max(0.0, rot_eigs[0]) /
                                 std::max(1e-12, rot_eigs[2]);
      last_info_min_per_measurement = std::max(0.0, full_eigs[0]) /
                                      std::max(1, n_meas);
      MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      auto &&HTz = H_sub_T * z;
      // K = K_1.block<DIM_STATE,6>(0,0) * H_sub_T;
      auto vec = (*state_propagat) - (*state);
      G.block<DIM_STATE, 7>(0, 0) = K_1.block<DIM_STATE, 7>(0, 0) * H_T_H.block<7, 7>(0, 0);
      MD(DIM_STATE, 1)
      solution = -K_1.block<DIM_STATE, 7>(0, 0) * HTz + vec - G.block<DIM_STATE, 7>(0, 0) * vec.block<7, 1>(0, 0);

      if (iteration == 0) {  // debug: raw prior-free GN solution (what the camera measurement wants before the prior), from VIO entry
        VD(DIM_STATE) raw_sol; raw_sol.setZero();
        raw_sol.block<7, 1>(0, 0) = (H_T_H.block<7, 7>(0, 0) + MD(7, 7)::Identity()).inverse() * HTz;
        StatesGroup tmp = (*state); tmp += raw_sol; raw_rot_vio_ = tmp.rot_end;
      }
      if (flip_roll || flip_pitch) {  // debug: flip VIO roll/pitch update in the WORLD frame (clean axis separation)
        V3D dw = state->rot_end * solution.block<3, 1>(0, 0);          // body-frame rot update -> world
        if (flip_roll) dw(0) = -dw(0);
        if (flip_pitch) dw(1) = -dw(1);
        solution.block<3, 1>(0, 0) = state->rot_end.transpose() * dw;  // world -> back to body
      }
      (*state) += solution;
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);

      auto &&expo_add = solution.block<1, 1>(6, 0);
      // if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f) && (expo_add.norm() < 0.001f)) EKF_end = true;
      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))  EKF_end = true;
    }
    else
    {
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    if (iteration == max_iterations || EKF_end) break;
  }
  // if (state->inv_expo_time < 0.0)  {ROS_ERROR("reset expo time!!!!!!!!!!\n"); state->inv_expo_time = 0.0;}
}

void VIOManager::updateFrameState(StatesGroup state)
{
  M3D Rwi(state.rot_end);
  V3D Pwi(state.pos_end);
  Rcw = Rci * Rwi.transpose();
  Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
  new_frame_->T_f_w_ = SE3(Rcw, Pcw);
}

void VIOManager::plotTrackedPoints()
{
  int total_points = visual_submap->voxel_points.size();
  if (total_points == 0) return;
  // int inlier_count = 0;
  // for (int i = 0; i < img_cp.rows / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Poaint2f(0, grid_size * i), cv::Point2f(img_cp.cols, grid_size * i), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.cols / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(grid_size * i, 0), cv::Point2f(grid_size * i, img_cp.rows), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.rows / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(0, grid_size * i), cv::Point2f(img_cp.cols, grid_size * i), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.cols / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(grid_size * i, 0), cv::Point2f(grid_size * i, img_cp.rows), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  for (int i = 0; i < total_points; i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];
    V2D pc(new_frame_->w2c(pt->pos_));

    if (visual_submap->errors[i] <= visual_submap->propa_errors[i])
    {
      // inlier_count++;
      cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(0, 255, 0), -1, 8); // Green Sparse Align tracked
    }
    else
    {
      cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(255, 0, 0), -1, 8); // Blue Sparse Align tracked
    }
  }
  // std::string text = std::to_string(inlier_count) + " " + std::to_string(total_points);
  // cv::Point2f origin;
  // origin.x = img_cp.cols - 110;
  // origin.y = 20;
  // cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, 8, 0);
}

V3F VIOManager::getInterpolatedPixel(cv::Mat img, V2D pc)
{
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int u_ref_i = floorf(pc[0]);
  const int v_ref_i = floorf(pc[1]);
  const float subpix_u_ref = (u_ref - u_ref_i);
  const float subpix_v_ref = (v_ref - v_ref_i);
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  uint8_t *img_ptr = (uint8_t *)img.data + ((v_ref_i)*width + (u_ref_i)) * 3;
  float B = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[0 + 3] + w_ref_bl * img_ptr[width * 3] + w_ref_br * img_ptr[width * 3 + 0 + 3];
  float G = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[1 + 3] + w_ref_bl * img_ptr[1 + width * 3] + w_ref_br * img_ptr[width * 3 + 1 + 3];
  float R = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[2 + 3] + w_ref_bl * img_ptr[2 + width * 3] + w_ref_br * img_ptr[width * 3 + 2 + 3];
  V3F pixel(B, G, R);
  return pixel;
}

void VIOManager::dumpDataForColmap()
{
  static int cnt = 1;
  std::ostringstream ss;
  ss << std::setw(5) << std::setfill('0') << cnt;
  std::string cnt_str = ss.str();
  std::string image_path = std::string(ROOT_DIR) + "Log/Colmap/images/" + cnt_str + ".png";
  
  cv::Mat img_rgb_undistort;
  pinhole_cam->undistortImage(img_rgb, img_rgb_undistort);
  cv::imwrite(image_path, img_rgb_undistort);
  
  Eigen::Quaterniond q(new_frame_->T_f_w_.rotation_matrix());
  Eigen::Vector3d t = new_frame_->T_f_w_.translation();
  fout_colmap << cnt << " "
            << std::fixed << std::setprecision(6)  // 保证浮点数精度为6位
            << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
            << t.x() << " " << t.y() << " " << t.z() << " "
            << 1 << " "  // CAMERA_ID (假设相机ID为1)
            << cnt_str << ".png" << std::endl;
  fout_colmap << "0.0 0.0 -1" << std::endl;
  cnt++;
}

void VIOManager::processFrame(
    cv::Mat &img, vector<pointWithVar> &pg,
    const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map,
    double img_time_s, double img_rel_s)
{
  last_translation_info_ratio = 0.0;
  last_rotation_info_ratio = 0.0;
  last_info_min_per_measurement = 0.0;
  if (width != img.cols || height != img.rows)
  {
    if (img.empty()) printf("[ VIO ] Empty Image!\n");
    cv::resize(img, img, cv::Size(img.cols * image_resize_factor, img.rows * image_resize_factor), 0, 0, CV_INTER_LINEAR);
  }
  img_rgb = img.clone();
  img_cp = img.clone();
  // img_test = img.clone();

  if (img.channels() == 3) cv::cvtColor(img, img, CV_BGR2GRAY);

  new_frame_.reset(new Frame(cam, img));
  updateFrameState(*state);
  
  resetGrid();

  double t1 = omp_get_wtime();

  retrieveFromVisualSparseMap(img, pg, feat_map);

  double t2 = omp_get_wtime();

  // The prior for this visual update is the post-propagation/post-LIO state at
  // VIO entry.  Snapshot it before covariance is updated so the diagnostic
  // logdet has the advertised prior-whitened meaning.
  if (visual_quality_log_enabled)
  {
    visual_quality_prior_state = *state;
    visual_quality_prior_state_cov = state->cov.block<7, 7>(0, 0);
  }

  if (state_update_enabled)
  {
    computeJacobianAndUpdateEKF(img);
  }
  else
  {
    // Tracking and visual-map maintenance below still run at the accepted LIO
    // pose.  Do not let a stale G from the previous frame touch covariance.
    G.setZero();
    updateFrameState(*state);
  }

  // The upstream visualizer already labels a patch as tracked when its final
  // photometric error is no worse than the propagated pose.  Preserve the
  // aggregate instead of throwing it away so fusion can assess measurement
  // quality without GT or a new image pass.
  int improved = 0;
  double final_error_sum = 0.0;
  double propagated_error_sum = 0.0;
  const size_t quality_count = std::min(visual_submap->errors.size(),
                                        visual_submap->propa_errors.size());
  for (size_t i = 0; i < quality_count; ++i)
  {
    const double final_error = visual_submap->errors[i];
    const double propagated_error = visual_submap->propa_errors[i];
    if (std::isfinite(final_error) && std::isfinite(propagated_error))
    {
      if (final_error <= propagated_error) ++improved;
      final_error_sum += final_error;
      propagated_error_sum += propagated_error;
    }
  }
  last_inlier_ratio = quality_count ? static_cast<double>(improved) / quality_count : 0.0;
  last_error_ratio = propagated_error_sum > 1e-9
      ? final_error_sum / propagated_error_sum : 1.0;

  // Re-evaluate the final optimizer level (level=0 plus each patch's stored
  // search level) residuals/Jacobians at the finalized accepted/rolled-back
  // state.  This mirrors updateState(img, 0) without applying another update.
  // This intentionally runs after the optimizer and is strictly read-only, so
  // a rejected final iteration can never leak its residual/Jacobian into the
  // CSV or affect estimator behavior.
  logVisualQualityFrame(img, img_time_s, img_rel_s);

  double t3 = omp_get_wtime();

  generateVisualMapPoints(img, pg);

  double t4 = omp_get_wtime();
  
  plotTrackedPoints();

  if (plot_flag) projectPatchFromRefToCur(feat_map);

  double t5 = omp_get_wtime();

  updateVisualMapPoints(img);

  double t6 = omp_get_wtime();

  updateReferencePatch(feat_map);

  double t7 = omp_get_wtime();
  
  if(colmap_output_en)  dumpDataForColmap();

  frame_count++;
  ave_total = ave_total * (frame_count - 1) / frame_count + (t7 - t1 - (t5 - t4)) / frame_count;

  // printf("[ VIO ] feat_map.size(): %zu\n", feat_map.size());
  // printf("\033[1;32m[ VIO time ]: current frame: retrieveFromVisualSparseMap time: %.6lf secs.\033[0m\n", t2 - t1);
  // printf("\033[1;32m[ VIO time ]: current frame: computeJacobianAndUpdateEKF time: %.6lf secs, comp H: %.6lf secs, ekf: %.6lf secs.\033[0m\n", t3 - t2, computeH, ekf_time);
  // printf("\033[1;32m[ VIO time ]: current frame: generateVisualMapPoints time: %.6lf secs.\033[0m\n", t4 - t3);
  // printf("\033[1;32m[ VIO time ]: current frame: updateVisualMapPoints time: %.6lf secs.\033[0m\n", t6 - t5);
  // printf("\033[1;32m[ VIO time ]: current frame: updateReferencePatch time: %.6lf secs.\033[0m\n", t7 - t6);
  // printf("\033[1;32m[ VIO time ]: current total time: %.6lf, average total time: %.6lf secs.\033[0m\n", t7 - t1 - (t5 - t4), ave_total);

  // ave_build_residual_time = ave_build_residual_time * (frame_count - 1) / frame_count + (t2 - t1) / frame_count;
  // ave_ekf_time = ave_ekf_time * (frame_count - 1) / frame_count + (t3 - t2) / frame_count;
 
  // cout << BLUE << "ave_build_residual_time: " << ave_build_residual_time << RESET << endl;
  // cout << BLUE << "ave_ekf_time: " << ave_ekf_time << RESET << endl;
  
  if (verbose)
  {
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         VIO Time                            |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Sparse Map Size", feat_map.size());
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "retrieveFromVisualSparseMap", t2 - t1);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "computeJacobianAndUpdateEKF", t3 - t2);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> computeJacobian", compute_jacobian_time);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> updateEKF", update_ekf_time);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "generateVisualMapPoints", t4 - t3);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateVisualMapPoints", t6 - t5);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateReferencePatch", t7 - t6);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Current Total Time", t7 - t1 - (t5 - t4));
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Average Total Time", ave_total);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  }

  // std::string text = std::to_string(int(1 / (t7 - t1 - (t5 - t4)))) + " HZ";
  // cv::Point2f origin;
  // origin.x = 20;
  // origin.y = 20;
  // cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, 8, 0);
  // cv::imwrite("/home/chunran/Desktop/raycasting/" + std::to_string(new_frame_->id_) + ".png", img_cp);
}
