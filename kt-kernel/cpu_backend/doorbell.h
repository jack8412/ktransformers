/**
 * Doorbell transport: replace the two cudaLaunchHostFunc nodes per MoE layer
 * with device-side value writes plus a wait, served by a spinning CPU poller.
 *
 * Measured motivation (Kimi-K3, 8xB200, 92 layers, frequency placement,
 * margin 10): ablating ONLY the two host-node dispatches moved decode from
 * 49.09 to 68.36 tok/s -- 62.4 us/layer, 77% of the 81 us residue that
 * remains after 7004e15's inline-empty check. The copies and merge are the
 * other 18.6 us and are not this transport's target.
 *
 * The host node is not slow because the callback is slow; it is slow because
 * reaching it serialises the stream on a driver-scheduled callback dispatch.
 * A value write is a memory operation performed in stream order, and the wait
 * is likewise a memop -- neither hands control to a driver thread.
 *
 * ---------------------------------------------------------------------------
 * PROTOCOL (arm / ring / wait)
 * ---------------------------------------------------------------------------
 * A captured graph node writes a CONSTANT: whatever value is recorded is the
 * value every replay writes. So a monotonic sequence CANNOT work here -- it
 * advances once at capture and never again, after which the poller's
 * "seq changed?" test is false forever and the GPU's "completion >= seq" wait
 * is already satisfied by the stale value. The CPU experts would silently
 * stop computing while the merge kept reading the first replay's output.
 * That failure is invisible in a smoke test, which is why the protocol is
 * built from constants that are correct under unlimited replay:
 *
 *     write completion[slot] = 0          (arm -- retract the last result)
 *     <staging D2H of activations + ids>
 *     write ring = slot + 1               (ring)
 *     wait  completion[slot] == slot + 1  (wait)
 *
 * and the poller, on seeing a non-zero ring:
 *
 *     v = ring.exchange(0)                (consume, so it fires once)
 *     work[v-1]()
 *     completion[v-1] = v                 (release)
 *
 * Every replay re-arms before it rings, so no stale completion can satisfy a
 * later wait. The ordering that makes it safe is stream order: replay N+1's
 * arm is recorded after replay N's wait, so the poller can never be mid-store
 * of completion when the arm lands.
 *
 * The ids/activation D2H MUST precede the ring. The poller's whole decision
 * (empty / full) reads those ids; if the ring became visible first it would
 * read the PREVIOUS step's batch and could declare a batch empty that is not.
 *
 * ---------------------------------------------------------------------------
 * ONE GLOBAL RING WORD
 * ---------------------------------------------------------------------------
 * The ring is a single word carrying the slot index, not a per-slot flag, so
 * the poller touches exactly two cache lines per event instead of sweeping
 * every slot. At 92 layers x ~52 captured batch-size tiers a per-slot sweep
 * would walk ~600 KB -- far past L2, and tens of microseconds, which is the
 * entire budget this transport is trying to win back.
 *
 * This is safe because at most ONE doorbell is ever outstanding: sglang gives
 * every KT layer the same CPU-side stream (get_stream("kt_cpu")), layers run
 * sequentially within a forward, and each layer's wait completes before the
 * next layer's arm is reached. `exchange` is used rather than load+store so a
 * ring that arrives mid-consume is not lost even if that invariant is ever
 * weakened.
 */
#ifndef CPUINFER_DOORBELL_H
#define CPUINFER_DOORBELL_H

#include <pthread.h>
#include <sched.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "./vendors/vendor.h"

// One full cache line per word: the ring is written by the device and read by
// the poller on every layer, and completion the other way round. Sharing a
// line would trade the host-node dispatch for coherence ping-pong.
static constexpr size_t KT_DB_STRIDE = 128;

class DoorbellTransport {
 public:
  static DoorbellTransport& instance() {
    static DoorbellTransport t;
    return t;
  }

  // Pinned + mapped so the device can write the ring and wait on completion
  // with stream memops while the poller reads/writes the same lines from the
  // host. Allocated ONCE, before any capture: the addresses are baked into
  // captured graphs, so they must never move -- and cudaHostAlloc is illegal
  // during capture, which is why slot binding (set_work) allocates nothing.
  void init(int num_slots, int num_pollers) {
    if (base_) return;
    if (num_slots <= 0) throw std::runtime_error("doorbell: num_slots must be > 0");
    n_slots_ = num_slots;
    // More than one poller cannot help: a single ring word means there is
    // never more than one event to serve. Extra threads would only contend on
    // that line. Kept as a parameter so the flag stays meaningful if the
    // protocol ever grows a per-socket ring.
    n_pollers_ = 1;
    // slot 0 of the page is the ring; completions start at stride 1.
    size_t bytes = KT_DB_STRIDE * (size_t)(num_slots + 1);
    void* p = nullptr;
    if (cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess)
      throw std::runtime_error("doorbell: cudaHostAlloc failed");
    std::memset(p, 0, bytes);
    base_ = (char*)p;
    void* dev = nullptr;
    if (cudaHostGetDevicePointer(&dev, p, 0) != cudaSuccess)
      throw std::runtime_error("doorbell: cudaHostGetDevicePointer failed");
    dev_base_ = (uintptr_t)dev;
    work_.resize(num_slots);
    ready_ = std::vector<std::atomic<uint8_t>>(num_slots);
    for (auto& r : ready_) r.store(0, std::memory_order_relaxed);
    printf("[doorbell] %d slots, %zu KB pinned, ring at host %p / dev 0x%llx\n", num_slots,
           bytes / 1024, (void*)base_, (unsigned long long)dev_base_);
    fflush(stdout);
  }

  bool inited() const { return base_ != nullptr; }
  int num_slots() const { return n_slots_; }

  // Device-mapped addresses handed to cuStreamWriteValue64 / cuStreamWaitValue64.
  uintptr_t ring_dev_addr() const { return dev_base_; }
  uintptr_t completion_dev_addr(int s) const {
    check_slot(s);
    return dev_base_ + (uintptr_t)(s + 1) * KT_DB_STRIDE;
  }

  // Bind a slot's work closure. Every pointer the closure captures (ring
  // buffers, the qlen cell) is stable for the life of that (layer, batch
  // size) pairing, and only the VALUES behind them change per step -- that is
  // what makes a doorbell enough: the device need only say "go", never what
  // to do. Allocates host memory only, so it is legal during graph capture.
  void set_work(int slot, std::function<void()> fn) {
    check_slot(slot);
    if (ready_[slot].load(std::memory_order_acquire))
      throw std::runtime_error("doorbell: slot " + std::to_string(slot) + " already bound");
    work_[slot] = std::move(fn);
    // Release pairs with the poller's acquire on ready_: without it the
    // poller could observe the slot as bound while the closure's captured
    // state is not yet visible to its core.
    ready_[slot].store(1, std::memory_order_release);
  }

  void start() {
    if (running_.load()) return;
    if (!base_) throw std::runtime_error("doorbell: start() before init()");
    running_.store(true);
    for (int t = 0; t < n_pollers_; ++t) pollers_.emplace_back([this, t] { poll(t); });
    printf("[doorbell] %d poller thread(s) started\n", n_pollers_);
    fflush(stdout);
  }

  void stop() {
    if (!running_.load()) return;
    running_.store(false);
    for (auto& th : pollers_)
      if (th.joinable()) th.join();
    pollers_.clear();
  }

  uint64_t served() const { return served_.load(std::memory_order_relaxed); }
  uint64_t spins() const { return spins_.load(std::memory_order_relaxed); }
  uint64_t unbound() const { return unbound_.load(std::memory_order_relaxed); }
  uint64_t work_ns_total() const { return work_ns_total_.load(std::memory_order_relaxed); }
  uint64_t work_ns_max() const { return work_ns_max_.load(std::memory_order_relaxed); }

  ~DoorbellTransport() { stop(); }

 private:
  DoorbellTransport() = default;

  void check_slot(int s) const {
    if (s < 0 || s >= n_slots_)
      throw std::runtime_error("doorbell: slot " + std::to_string(s) + " out of range (" +
                               std::to_string(n_slots_) + " slots)");
  }

  std::atomic<uint64_t>* ring_atomic() const {
    return reinterpret_cast<std::atomic<uint64_t>*>(base_);
  }
  std::atomic<uint64_t>* completion_atomic(int s) const {
    return reinterpret_cast<std::atomic<uint64_t>*>(base_ + (size_t)(s + 1) * KT_DB_STRIDE);
  }

  void poll(int) {
    // Optional pinning: KT_DOORBELL_POLLER_CPU=<n>. No default -- the
    // transport band on this box has never been measured (sysstat is not
    // installed), and pinning to a core a compute worker also owns would
    // convert the design's advantage into a latency lottery. Better to leave
    // placement to the scheduler than to assert a band we have not verified.
    if (const char* cpu = std::getenv("KT_DOORBELL_POLLER_CPU")) {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(atoi(cpu), &set);
      if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0)
        printf("[doorbell] poller pinned to cpu %s\n", cpu);
      else
        printf("[doorbell] WARNING: could not pin poller to cpu %s\n", cpu);
      fflush(stdout);
    }

    auto* ring = ring_atomic();
    auto last_work = std::chrono::steady_clock::now();
    // Spin hot while a decode is in flight; back off only after an idle gap
    // far longer than any inter-layer or inter-step gap (a 40 tok/s step is
    // ~25 ms), so the backoff can only be paid between requests.
    constexpr auto kIdleBeforeSleep = std::chrono::milliseconds(50);

    while (running_.load(std::memory_order_relaxed)) {
      // Acquire: the device's ring write publishes the ids and activations
      // copied before it. Reading them before this load would be reading the
      // previous step's batch. Exchange (not load+store) so a ring arriving
      // mid-consume cannot be dropped.
      uint64_t v = ring->exchange(0, std::memory_order_acquire);
      if (v == 0) {
        spins_.fetch_add(1, std::memory_order_relaxed);
#if defined(__x86_64__)
        __builtin_ia32_pause();
#endif
        if (std::chrono::steady_clock::now() - last_work > kIdleBeforeSleep)
          std::this_thread::sleep_for(std::chrono::microseconds(20));
        continue;
      }

      int s = (int)(v - 1);
      if (s < 0 || s >= n_slots_) {
        // Cannot publish a completion for a slot we cannot name; the GPU that
        // rang it will hang, which is the honest outcome for a corrupt ring.
        unbound_.fetch_add(1, std::memory_order_relaxed);
        printf("[doorbell] ERROR: ring value %llu out of range\n", (unsigned long long)v);
        fflush(stdout);
        continue;
      }

      if (ready_[s].load(std::memory_order_acquire)) {
        auto t0 = std::chrono::steady_clock::now();
        work_[s]();
        uint64_t ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - t0).count();
        work_ns_total_.fetch_add(ns, std::memory_order_relaxed);
        uint64_t prev = work_ns_max_.load(std::memory_order_relaxed);
        while (ns > prev && !work_ns_max_.compare_exchange_weak(prev, ns, std::memory_order_relaxed)) {
        }
      } else {
        // Rung but never bound. Completing anyway would hand the merge a
        // stale output buffer -- silently wrong numbers, the exact failure
        // this protocol exists to avoid. Count it, say so, and let the wait
        // hang: a hang is diagnosable, a wrong logprob is not.
        unbound_.fetch_add(1, std::memory_order_relaxed);
        printf("[doorbell] ERROR: slot %d rung before it was bound\n", s);
        fflush(stdout);
        continue;
      }

      // Release: the output buffer must be visible before the GPU's wait is
      // satisfied, or the H2D copy races the result.
      completion_atomic(s)->store((uint64_t)s + 1, std::memory_order_release);
      served_.fetch_add(1, std::memory_order_relaxed);
      last_work = std::chrono::steady_clock::now();
    }
  }

  char* base_ = nullptr;
  uintptr_t dev_base_ = 0;
  int n_slots_ = 0;
  int n_pollers_ = 1;
  std::vector<std::function<void()>> work_;
  std::vector<std::atomic<uint8_t>> ready_;
  std::vector<std::thread> pollers_;
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> served_{0};
  std::atomic<uint64_t> spins_{0};
  std::atomic<uint64_t> unbound_{0};
  std::atomic<uint64_t> work_ns_total_{0};
  std::atomic<uint64_t> work_ns_max_{0};
};

#endif  // CPUINFER_DOORBELL_H
