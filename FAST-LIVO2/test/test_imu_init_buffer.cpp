#include "imu_init_buffer.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Sample
{
  std::uint64_t stamp_ns;
  std::uint32_t seq;
};

using Buffer = fast_livo::ImuInitBuffer<Sample>;

void require(const bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

std::vector<std::uint32_t> runPartition(const std::vector<std::size_t> &batches)
{
  constexpr std::uint64_t anchor = 1'000'000'000ULL;
  std::vector<Sample> stream;
  stream.push_back({anchor - 5, 99});
  stream.push_back({anchor, 100});
  for (std::uint32_t index = 1; index <= 40; ++index)
    stream.push_back({anchor + index * 5, 100 + index});

  Buffer buffer(128, 20);
  require(buffer.setExplicitAnchor(anchor), "explicit anchor rejected");
  std::vector<std::uint32_t> selected;
  std::size_t cursor = 0;
  for (const std::size_t batch_size : batches)
  {
    for (std::size_t count = 0; count < batch_size && cursor < stream.size(); ++count)
    {
      const Sample sample = stream[cursor++];
      require(buffer.push(sample, sample.stamp_ns), "ordered sample rejected");
    }
    const std::uint64_t epoch = cursor == 0 ? anchor : stream[cursor - 1].stamp_ns;
    auto drained = buffer.takeThrough(epoch);
    require(!drained.failed, "partition drain failed: " + drained.reason);
    for (const Sample &sample : drained.samples)
      selected.push_back(sample.seq);
  }
  while (cursor < stream.size())
  {
    const Sample sample = stream[cursor++];
    require(buffer.push(sample, sample.stamp_ns), "tail sample rejected");
  }
  auto drained = buffer.takeThrough(stream.back().stamp_ns);
  require(!drained.failed, "tail drain failed: " + drained.reason);
  for (const Sample &sample : drained.samples)
    selected.push_back(sample.seq);
  return selected;
}

void testPartitionIndependence()
{
  const auto one_by_one = runPartition(std::vector<std::size_t>(42, 1));
  const auto callback_bursts = runPartition({2, 13, 1, 9, 17});
  const auto one_batch = runPartition({42});
  require(one_by_one == callback_bursts, "burst partition changed selection");
  require(one_by_one == one_batch, "single batch changed selection");
  require(one_by_one.size() == 40, "wrong selected size");
  require(one_by_one.front() == 101, "strict anchor boundary not enforced");
  require(one_by_one.at(29) == 130, "first 30 sequence changed");
}

void testLiveAnchorUsesBufferedPrefixAndStrictBoundary()
{
  Buffer buffer(16, 20);
  const Sample before{995, 1};
  const Sample equal{1000, 2};
  const Sample after{1001, 3};
  require(buffer.push(before, before.stamp_ns), "live before rejected");
  require(buffer.push(equal, equal.stamp_ns), "live equal rejected");
  require(buffer.latchLiveAnchor(1000), "live anchor rejected");
  require(buffer.push(after, after.stamp_ns), "live after rejected");
  auto drained = buffer.takeThrough(1001);
  require(!drained.failed, "live drain failed");
  require(drained.samples.size() == 1, "live strict interval size wrong");
  require(drained.samples.front().seq == 3, "anchor-equal sample included");
}

void testExplicitAnchorPredecessorContract()
{
  {
    Buffer buffer(8, 20);
    require(buffer.setExplicitAnchor(1000), "explicit anchor rejected");
    require(buffer.push(Sample{1001, 3}, 1001), "post-anchor sample rejected early");
    auto drained = buffer.takeThrough(1001);
    require(drained.failed, "missing predecessor did not fail");
  }
  {
    Buffer buffer(8, 20);
    require(buffer.setExplicitAnchor(1000), "explicit anchor rejected");
    require(buffer.push(Sample{979, 1}, 979), "stale predecessor rejected early");
    require(buffer.push(Sample{1001, 3}, 1001), "post-anchor sample rejected early");
    auto drained = buffer.takeThrough(1001);
    require(drained.failed, "predecessor gap above limit did not fail");
  }
  {
    Buffer buffer(8, 20);
    require(buffer.setExplicitAnchor(1000), "explicit anchor rejected");
    require(buffer.push(Sample{980, 1}, 980), "boundary predecessor rejected");
    require(buffer.push(Sample{1001, 3}, 1001), "post-anchor sample rejected");
    auto drained = buffer.takeThrough(1001);
    require(!drained.failed, "predecessor exactly at limit failed");
    require(drained.samples.size() == 1, "valid predecessor drain wrong");
    require(buffer.anchorPredecessorStampNs() == 980,
            "wrong predecessor stamp retained");
  }
}

void testDuplicateAndBackwardFailClosed()
{
  for (const Sample bad : {Sample{1000, 2}, Sample{999, 3}})
  {
    Buffer buffer(8, 20);
    require(buffer.push(Sample{1000, 1}, 1000), "first sample rejected");
    require(!buffer.push(bad, bad.stamp_ns),
            "duplicate/backward sample did not fail");
    require(buffer.failed(), "duplicate/backward failure not latched");
    auto drained = buffer.takeThrough(1001);
    require(drained.failed, "failed state did not propagate through drain");
  }
}

void testRejectedWindowContinuationAndLegacySuffixFixture()
{
  // Buffer emits every post-anchor sample exactly once across sync epochs. An
  // initializer rejecting [101..130] can therefore consume [131..160] next,
  // without overlap or callback-partition dependence. Runtime continues to
  // define state at the synchronized image epoch (1200 here), not sample 160.
  Buffer buffer(128, 20);
  require(buffer.setExplicitAnchor(1000), "explicit anchor rejected");
  require(buffer.push(Sample{995, 99}, 995), "predecessor rejected");
  for (std::uint32_t index = 1; index <= 60; ++index)
    require(buffer.push(Sample{1000 + index, 100 + index}, 1000 + index),
            "window sample rejected");
  auto first = buffer.takeThrough(1030);
  auto second = buffer.takeThrough(1060);
  require(first.samples.size() == 30, "first rejected-window fixture wrong");
  require(second.samples.size() == 30, "second window fixture wrong");
  require(first.samples.front().seq == 101 && first.samples.back().seq == 130,
          "first window endpoints wrong");
  require(second.samples.front().seq == 131 && second.samples.back().seq == 160,
          "rejected window overlapped or skipped");
  constexpr std::uint64_t legacy_state_epoch = 1200;
  require(legacy_state_epoch > second.samples.back().stamp_ns,
          "legacy state epoch fixture must remain the later sync epoch");
}

void testOverflowFailsClosed()
{
  Buffer buffer(2, 20);
  require(buffer.push(Sample{1, 1}, 1), "overflow fixture first rejected");
  require(buffer.push(Sample{2, 2}, 2), "overflow fixture second rejected");
  require(!buffer.push(Sample{3, 3}, 3), "overflow did not fail");
  require(buffer.failed(), "overflow failure not latched");
  require(buffer.dropCount() == 1, "overflow drop count wrong");
}

void testExactParser()
{
  require(fast_livo::parsePositiveNanoseconds("1785863998853296757") ==
              1785863998853296757ULL,
          "uint64 timestamp parser rounded input");
  for (const std::string bad : {"", "0", "-1", "1.5", "18446744073709551616"})
  {
    bool threw = false;
    try
    {
      static_cast<void>(fast_livo::parsePositiveNanoseconds(bad));
    }
    catch (const std::invalid_argument &)
    {
      threw = true;
    }
    require(threw, "invalid nanosecond string accepted: " + bad);
  }
}

} // namespace

int main()
{
  try
  {
    testPartitionIndependence();
    testLiveAnchorUsesBufferedPrefixAndStrictBoundary();
    testExplicitAnchorPredecessorContract();
    testDuplicateAndBackwardFailClosed();
    testRejectedWindowContinuationAndLegacySuffixFixture();
    testOverflowFailsClosed();
    testExactParser();
    std::cout << "imu_init_buffer tests passed\n";
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << "imu_init_buffer test failure: " << error.what() << '\n';
    return 1;
  }
}
