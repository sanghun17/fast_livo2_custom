#include "IMU_Processing.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(const bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

sensor_msgs::Imu::ConstPtr makeImu(
    const std::uint64_t stamp_ns, const std::uint32_t seq,
    const V3D &acceleration, const V3D &angular_velocity)
{
  sensor_msgs::Imu::Ptr sample(new sensor_msgs::Imu());
  sample->header.stamp.fromNSec(stamp_ns);
  sample->header.seq = seq;
  sample->linear_acceleration.x = acceleration.x();
  sample->linear_acceleration.y = acceleration.y();
  sample->linear_acceleration.z = acceleration.z();
  sample->angular_velocity.x = angular_velocity.x();
  sample->angular_velocity.y = angular_velocity.y();
  sample->angular_velocity.z = angular_velocity.z();
  return sample;
}

LidarMeasureGroup makeGroup(
    const double synchronized_epoch,
    const sensor_msgs::Imu::ConstPtr &legacy_last_sample)
{
  LidarMeasureGroup group;
  group.lio_vio_flg = LIO;
  MeasureGroup measurement;
  measurement.lio_time = synchronized_epoch;
  measurement.imu.push_back(legacy_last_sample);
  group.measures.push_back(measurement);
  return group;
}

void testRejectedWindowResetsAndNextExactWindowInitializes()
{
  ImuProcess process;
  process.set_imu_init_frame_num(3);
  process.set_imu_init_stationarity(0.30, 0.25, 1.50, 3.00, false);

  constexpr std::uint64_t base_ns = 1'000'000'000ULL;
  constexpr std::uint64_t state_epoch_ns = 2'000'000'000ULL;
  const auto legacy_last = makeImu(
      1'900'000'000ULL, 900, V3D(0.0, 0.0, G_m_s2), V3D::Zero());
  LidarMeasureGroup group = makeGroup(2.0, legacy_last);
  StatesGroup state;
  PointCloudXYZI::Ptr output(new PointCloudXYZI());

  deque<sensor_msgs::Imu::ConstPtr> moving_window;
  for (std::uint32_t index = 1; index <= 3; ++index)
    moving_window.push_back(makeImu(
        base_ns + index * 5'000'000ULL, 100 + index,
        V3D(0.0, 0.0, G_m_s2), V3D(1.0, 0.0, 0.0)));
  process.Process2(group, state, output, &moving_window);

  require(process.imu_need_init,
          "non-stationary window unexpectedly initialized the estimator");
  require(process.imu_init_rejected_windows() == 1,
          "rejected-window counter did not increment");
  require(process.imu_init_sample_count() == 0,
          "rejected window did not reset the sample count");
  require(process.imu_init_selected_samples().empty(),
          "rejected window left stale selected stamps");

  deque<sensor_msgs::Imu::ConstPtr> stationary_window;
  for (std::uint32_t index = 1; index <= 3; ++index)
    stationary_window.push_back(makeImu(
        base_ns + (3 + index) * 5'000'000ULL, 103 + index,
        V3D(0.0, 0.0, G_m_s2), V3D::Zero()));
  process.Process2(group, state, output, &stationary_window);

  require(!process.imu_need_init,
          "stationary replacement window did not initialize the estimator");
  require(process.imu_init_rejected_windows() == 1,
          "accepted window lost rejection provenance");
  require(process.imu_init_sample_count() == 3,
          "accepted window did not contain exactly three samples");
  const auto &selected = process.imu_init_selected_samples();
  require(selected.size() == 3, "selected vector size is not exact");
  require(selected.front().first == base_ns + 20'000'000ULL &&
              selected.front().second == 104,
          "accepted vector retained a sample from the rejected window");
  require(selected.back().first == base_ns + 30'000'000ULL &&
              selected.back().second == 106,
          "accepted vector endpoint is wrong");
  require(std::abs(process.imu_last_prop_end_time() -
                   static_cast<double>(state_epoch_ns) * 1e-9) < 1e-12,
          "legacy synchronized state epoch changed");
  require(process.imu_last_sample_stamp_ns() ==
              legacy_last->header.stamp.toNSec(),
          "legacy MeasureGroup last-IMU/suffix semantics changed");
  require((state.gravity - V3D(0.0, 0.0, -G_m_s2)).norm() < 1e-12,
          "accepted stationary window produced the wrong gravity");
  require(state.bias_g.isZero(0.0),
          "gyro-bias estimation changed despite being disabled");

  process.Reset();
  require(process.imu_need_init, "Reset did not re-arm initialization");
  require(process.imu_init_sample_count() == 0,
          "Reset retained initialization samples");
  require(process.imu_init_selected_samples().empty(),
          "Reset retained the selected sample vector");
  require(process.imu_init_rejected_windows() == 0,
          "Reset retained prior-attempt rejection diagnostics");
  require(process.imu_init_invalid_samples() == 0,
          "Reset retained prior-attempt invalid-sample diagnostics");
  require(process.imu_last_prop_end_time() == 0.0,
          "Reset retained the prior synchronized epoch");
  require(process.imu_last_sample_stamp_ns() == 0,
          "Reset retained the prior MeasureGroup last IMU");
}

} // namespace

int main()
{
  // ROS_*_THROTTLE diagnostics consult ROS time even in this standalone unit
  // test.  Initialize only the time subsystem; no master or subscriptions are
  // involved.
  ros::Time::init();
  try
  {
    testRejectedWindowResetsAndNextExactWindowInitializes();
    std::cout << "imu initialization integration tests passed\n";
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << "imu initialization integration test failure: "
              << error.what() << '\n';
    return 1;
  }
}
