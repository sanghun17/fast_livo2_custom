#ifndef FAST_LIVO_IMU_INIT_BUFFER_H
#define FAST_LIVO_IMU_INIT_BUFFER_H

#include <cstdint>
#include <deque>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fast_livo {

inline std::uint64_t parsePositiveNanoseconds(const std::string &text)
{
  if (text.empty()) throw std::invalid_argument("empty nanosecond timestamp");

  std::uint64_t value = 0;
  for (const char character : text)
  {
    if (character < '0' || character > '9')
      throw std::invalid_argument("nanosecond timestamp must contain decimal digits only");
    const std::uint64_t digit = static_cast<std::uint64_t>(character - '0');
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U)
      throw std::invalid_argument("nanosecond timestamp overflows uint64");
    value = value * 10U + digit;
  }
  if (value == 0) throw std::invalid_argument("nanosecond timestamp must be positive");
  return value;
}

/** A bounded, sensor-time selector used only while IMU initialization is pending.
 *
 * The normal synchronizer is free to consume its own IMU queue.  This buffer
 * retains the same callback objects independently and emits each sample once,
 * in the strict sensor-time interval (anchor, synchronized_epoch].  It does
 * not decide sample validity or stationarity; ImuProcess remains the single
 * owner of those policies.
 */
template <typename Sample>
class ImuInitBuffer
{
public:
  struct DrainResult
  {
    std::deque<Sample> samples;
    bool failed = false;
    std::string reason;
  };

  explicit ImuInitBuffer(
      const std::size_t maximum_size,
      const std::uint64_t maximum_anchor_predecessor_gap_ns = 0)
      : maximum_size_(maximum_size),
        maximum_anchor_predecessor_gap_ns_(maximum_anchor_predecessor_gap_ns)
  {
    if (maximum_size_ == 0)
      throw std::invalid_argument("IMU initialization buffer size must be positive");
  }

  bool setExplicitAnchor(const std::uint64_t anchor_stamp_ns)
  {
    return setAnchor(anchor_stamp_ns, true);
  }

  bool latchLiveAnchor(const std::uint64_t anchor_stamp_ns)
  {
    return setAnchor(anchor_stamp_ns, false);
  }

  bool push(const Sample &sample, const std::uint64_t stamp_ns)
  {
    if (complete_ || failed_) return !failed_;
    if (stamp_ns == 0)
    {
      fail("initialization sample has a zero timestamp");
      return false;
    }
    if (last_pushed_stamp_ns_ != 0 && stamp_ns <= last_pushed_stamp_ns_)
    {
      std::ostringstream message;
      message << "non-increasing initialization sample timestamp " << stamp_ns
              << " after " << last_pushed_stamp_ns_;
      fail(message.str());
      return false;
    }
    if (queue_.size() >= maximum_size_)
    {
      fail("initialization queue overflow");
      ++drop_count_;
      return false;
    }
    queue_.push_back({stamp_ns, sample});
    last_pushed_stamp_ns_ = stamp_ns;
    high_water_ = queue_.size() > high_water_ ? queue_.size() : high_water_;
    if (anchor_latched_ && stamp_ns <= anchor_stamp_ns_)
    {
      anchor_has_preceding_imu_ = true;
      anchor_predecessor_stamp_ns_ = stamp_ns;
      ++at_or_before_anchor_count_;
    }
    return true;
  }

  DrainResult takeThrough(const std::uint64_t synchronized_epoch_ns)
  {
    DrainResult result;
    if (failed_)
    {
      result.failed = true;
      result.reason = failure_reason_;
      return result;
    }
    if (complete_ || !anchor_latched_ || synchronized_epoch_ns <= anchor_stamp_ns_)
      return result;

    // An explicit offline anchor is meaningful only if this subscriber saw
    // the IMU prefix bracketing it.  Otherwise the first post-anchor sample may
    // have been lost during transport startup and silently sliding is unsafe.
    if (explicit_anchor_ && !anchorHasPredecessorWithinLimit())
    {
      fail("explicit anchor has no received IMU predecessor within the configured maximum gap");
      result.failed = true;
      result.reason = failure_reason_;
      return result;
    }

    while (!queue_.empty() && queue_.front().stamp_ns <= anchor_stamp_ns_)
    {
      queue_.pop_front();
      ++discarded_at_or_before_anchor_count_;
    }
    while (!queue_.empty() && queue_.front().stamp_ns <= synchronized_epoch_ns)
    {
      result.samples.push_back(queue_.front().sample);
      queue_.pop_front();
      ++emitted_count_;
    }
    return result;
  }

  void markComplete()
  {
    complete_ = true;
    queue_.clear();
  }

  bool anchorLatched() const { return anchor_latched_; }
  bool explicitAnchor() const { return explicit_anchor_; }
  bool anchorHasPrecedingImu() const { return anchor_has_preceding_imu_; }
  bool anchorHasPredecessorWithinLimit() const
  {
    if (!anchor_has_preceding_imu_) return false;
    if (maximum_anchor_predecessor_gap_ns_ == 0) return true;
    return anchor_stamp_ns_ >= anchor_predecessor_stamp_ns_ &&
           anchor_stamp_ns_ - anchor_predecessor_stamp_ns_ <=
               maximum_anchor_predecessor_gap_ns_;
  }
  bool failed() const { return failed_; }
  bool complete() const { return complete_; }
  std::uint64_t anchorStampNs() const { return anchor_stamp_ns_; }
  std::size_t size() const { return queue_.size(); }
  std::size_t highWater() const { return high_water_; }
  std::uint64_t dropCount() const { return drop_count_; }
  std::uint64_t atOrBeforeAnchorCount() const { return at_or_before_anchor_count_; }
  std::uint64_t anchorPredecessorStampNs() const
  {
    return anchor_predecessor_stamp_ns_;
  }
  std::uint64_t discardedAtOrBeforeAnchorCount() const
  {
    return discarded_at_or_before_anchor_count_;
  }
  std::uint64_t emittedCount() const { return emitted_count_; }
  const std::string &failureReason() const { return failure_reason_; }

private:
  struct Entry
  {
    std::uint64_t stamp_ns;
    Sample sample;
  };

  bool setAnchor(const std::uint64_t anchor_stamp_ns, const bool explicit_anchor)
  {
    if (anchor_stamp_ns == 0)
    {
      fail("anchor timestamp must be positive");
      return false;
    }
    if (anchor_latched_)
    {
      if (anchor_stamp_ns_ != anchor_stamp_ns || explicit_anchor_ != explicit_anchor)
      {
        fail("attempted to change an already latched initialization anchor");
        return false;
      }
      return true;
    }

    anchor_stamp_ns_ = anchor_stamp_ns;
    anchor_latched_ = true;
    explicit_anchor_ = explicit_anchor;
    for (const Entry &entry : queue_)
    {
      if (entry.stamp_ns <= anchor_stamp_ns_)
      {
        anchor_has_preceding_imu_ = true;
        anchor_predecessor_stamp_ns_ = entry.stamp_ns;
        ++at_or_before_anchor_count_;
      }
    }
    return true;
  }

  void fail(const std::string &reason)
  {
    failed_ = true;
    failure_reason_ = reason;
  }

  std::size_t maximum_size_;
  std::uint64_t maximum_anchor_predecessor_gap_ns_ = 0;
  std::deque<Entry> queue_;
  std::uint64_t anchor_stamp_ns_ = 0;
  std::size_t high_water_ = 0;
  std::uint64_t drop_count_ = 0;
  std::uint64_t at_or_before_anchor_count_ = 0;
  std::uint64_t discarded_at_or_before_anchor_count_ = 0;
  std::uint64_t emitted_count_ = 0;
  std::uint64_t last_pushed_stamp_ns_ = 0;
  std::uint64_t anchor_predecessor_stamp_ns_ = 0;
  bool anchor_latched_ = false;
  bool explicit_anchor_ = false;
  bool anchor_has_preceding_imu_ = false;
  bool failed_ = false;
  bool complete_ = false;
  std::string failure_reason_;
};

} // namespace fast_livo

#endif
