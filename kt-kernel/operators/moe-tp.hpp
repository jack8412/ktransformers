#ifndef CPUINFER_OPERATOR_MOE_HPP
#define CPUINFER_OPERATOR_MOE_HPP

// #define CHECK

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>

#include "../cpu_backend/shared_mem_buffer.h"
#include "common.hpp"

/// Partitions whose rows one (token, slot) can be spread over. Sized well
/// above any real NUMA count; a partition past it is dropped rather than
/// overrunning the stack array.
static constexpr int kReapMaxPartitions = 16;

/// bf16 -> fp32 is the bit pattern shifted into the high half.
static inline float kt_bf16_to_fp32(ggml_bf16_t v) {
  uint32_t bits = static_cast<uint32_t>(v.bits) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

// Forward declaration for Llamafile backend type checking
class LLAMA_MOE_TP;

template <typename T>
concept MOE_TP_PART = requires(T t, int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                               void* output, GeneralMOEConfig config, int tp_idx) {
  typename T::output_t;
  { new T(config, tp_idx) } -> std::same_as<T*>;
  { t.forward(qlen, k, expert_ids, weights, input, output) } -> std::same_as<void>;
  // { t.load_weights() } -> std::same_as<void>;
};

template <MOE_TP_PART T, typename Concrete = T>
class TP_MOE_Common : public MoE_Interface {
  static_assert(std::is_base_of_v<T, Concrete>);
  static_assert(std::is_constructible_v<Concrete, GeneralMOEConfig, int>);

 protected:
  std::vector<GeneralMOEConfig> tp_configs;
  int tp_count;
  int me_numa_id;
  std::vector<std::unique_ptr<T>> tps;

  std::vector<typename T::output_t*> local_output_numa;
  typename T::output_t* local_output = nullptr;

  bool weights_loaded = false;

#ifdef FORWARD_TIME_REPORT
  size_t forward_time_sum_ns = 0;
  size_t forward_count = 0;
#endif
 public:
  GeneralMOEConfig config;
  // Count of forwards completed inline because the batch routed entirely to
  // GPU-resident experts (see the forward binding's host callback). Written
  // from the CUDA host-callback thread, read from Python for telemetry.
  std::atomic<uint64_t> inline_empty_forwards{0};
  // REAP scoring: one norm per (token, slot) this layer computed, written
  // straight into a pinned host buffer the caller owns. Norms only -- the
  // weights and expert ids that turn one into a score live on the caller's
  // side, and so does the knowledge of which rows of a padded decode batch
  // are real.
  float* reap_norms_ = nullptr;
  int reap_max_tokens_ = 0;
  int reap_top_k_ = 0;
  using input_t = typename T::input_t;
  TP_MOE_Common(const GeneralMOEConfig& config) : config(config) {
    printf("TP MOE layer %d, pool: 0x%lx, expert num: %d, num_experts_per_tok: %d\n", config.layer_idx,
           (intptr_t)config.pool, config.expert_num, config.num_experts_per_tok);
    if (config.pool == nullptr) {
      printf("TP MOE layer %d, no worker pool\n", config.layer_idx);
      throw std::runtime_error("no worker pool");
    }

    this->config = config;
    tp_count = config.pool->config.subpool_count;
    if (config.intermediate_size % tp_count != 0) {
      printf("intermediate_size %d, tp count %d\n", config.intermediate_size, tp_count);
      throw std::runtime_error(
          "For TP, intermediate_size must be a "
          "multiple of NUMA node count");
    }

    // Check if this is Llamafile backend using compile-time type checking
    constexpr bool is_llamafile = std::is_same<T, LLAMA_MOE_TP>::value;
#ifndef QK_K
#define QK_K 256
#endif

    if (is_llamafile) {
      // For Llamafile backend: use QK_K-aligned TP splitting
      if (config.intermediate_size % QK_K != 0) {
        printf("intermediate_size %d must be divisible by QK_K %d for Llamafile backend\n", config.intermediate_size,
               QK_K);
        throw std::runtime_error("intermediate_size must be divisible by QK_K (256) for Llamafile backend");
      }

      int num_blocks = config.intermediate_size / QK_K;
      int base_blocks = num_blocks / tp_count;
      int extra_blocks = num_blocks % tp_count;

      if (base_blocks == 0) {
        printf("intermediate_size %d is too small for tp_count %d (num_blocks=%d)\n", config.intermediate_size,
               tp_count, num_blocks);
        throw std::runtime_error("intermediate_size too small: cannot distribute blocks to all TP instances");
      }

      printf("Llamafile TP splitting: intermediate_size=%d, tp_count=%d, QK_K=%d\n", config.intermediate_size, tp_count,
             QK_K);
      printf("  num_blocks=%d, base_blocks=%d, extra_blocks=%d\n", num_blocks, base_blocks, extra_blocks);

      int current_offset = 0;
      for (auto i = 0; i < tp_count; i++) {
        tps.push_back(nullptr);
        GeneralMOEConfig tp_config = config;

        // First extra_blocks TPs get one more block
        int num_blocks_for_this_tp = base_blocks + (i < extra_blocks ? 1 : 0);
        tp_config.intermediate_size = num_blocks_for_this_tp * QK_K;

        printf("  TP %d: intermediate_size=%d, offset=%d, blocks=%d\n", i, tp_config.intermediate_size, current_offset,
               num_blocks_for_this_tp);

        tp_configs.push_back(tp_config);
        current_offset += tp_config.intermediate_size;
      }
    } else {
      // For non-Llamafile backends: use simple equal division
      if (config.intermediate_size % tp_count != 0) {
        printf("intermediate_size %d, tp count %d\n", config.intermediate_size, tp_count);
        throw std::runtime_error(
            "For TP, intermediate_size must be a "
            "multiple of NUMA node count");
      }

      for (auto i = 0; i < tp_count; i++) {
        tps.push_back(nullptr);
        GeneralMOEConfig tp_config = config;
        tp_config.intermediate_size /= tp_count;
        tp_configs.push_back(tp_config);
      }
    }

    config.pool->dispense_backend()->do_numa_job(
        [this, config](int i) { tps[i] = std::unique_ptr<T>(new Concrete(tp_configs[i], i)); });

    local_output_numa.resize(tp_count, nullptr);
    MemoryRequest mem_requests;
    for (auto i = 0; i < tp_count; i++) {
      mem_requests.append_pointer(
          &local_output_numa[i],
          (size_t)sizeof(typename T::output_t) * tp_configs[i].max_possible_qlen() * tp_configs[i].hidden_size);
    }
    mem_requests.append_pointer(
        (void**)&local_output,
        sizeof(typename T::output_t) * tp_configs[0].max_possible_qlen() * tp_configs[0].hidden_size);
    // printf("local output tp, %d,\n", tp_configs[0].max_possible_qlen());
    shared_mem_buffer.alloc(this, mem_requests);
  }

  virtual ~TP_MOE_Common() {
    // Unregister local_output requests; see SharedMemBuffer::dealloc.
    shared_mem_buffer.dealloc(this);
  }

  void warm_up() {
    int qlen = config.max_possible_qlen();
    std::vector<uint8_t> input(sizeof(ggml_bf16_t) * qlen * config.hidden_size);
    std::vector<uint8_t> output(sizeof(ggml_bf16_t) * qlen * config.hidden_size);
    std::vector<int64_t> expert_ids(qlen * config.num_experts_per_tok);
    std::vector<float> weights(qlen * config.num_experts_per_tok);
    for (int i = 0; i < qlen * config.num_experts_per_tok; i++) {
      expert_ids[i] = i % config.expert_num;
      weights[i] = 0.01;
    }
    forward(&qlen, config.num_experts_per_tok, expert_ids.data(), weights.data(), input.data(), output.data(), false);
  }

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output,
               bool incremental = false) {
    int qlen_local = qlen;
    forward(&qlen_local, k, expert_ids, weights, input, output, incremental);
  }

  void forward(int* qlen_ptr, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    forward(qlen_ptr, k, expert_ids, weights, input, output, false);
  }

  void forward_binding(intptr_t qlen_ptr, int k, intptr_t expert_ids, intptr_t weights, intptr_t input, intptr_t output,
                       bool incremental) {
    forward((int*)qlen_ptr, k, (const int64_t*)expert_ids, (const float*)weights, (const void*)input, (void*)output,
            incremental);
  }

  // True if any routed slot in this batch names an expert this CPU instance
  // owns.  Used by the forward binding's host callback to decide whether the
  // forward is worth waking the worker thread for: under GPU-preferred
  // (margin) routing almost every layer-step routes entirely to GPU-resident
  // experts, and the enqueue -> wake -> run -> signal round trip then costs
  // far more than the empty work it carries.  Cheap: qlen*k mask lookups,
  // no allocation, no locks.
  bool batch_has_cpu_expert(int qlen, int k, const int64_t* expert_ids) const {
    if (expert_ids == nullptr || config.gpu_experts_mask == nullptr) {
      return true;  // nothing to prove it is empty -> take the normal path
    }
    const long total = (long)qlen * (long)k;
    for (long i = 0; i < total; ++i) {
      if (!config.should_skip_expert(expert_ids[i])) {
        return true;
      }
    }
    return false;
  }

  // Bytes a forward writes into the caller's output buffer: [qlen, hidden]
  // bf16, matching the ggml_bf16_t stores in merge_results().
  size_t output_bytes(int qlen) const { return (size_t)qlen * (size_t)config.hidden_size * sizeof(uint16_t); }

  /// \brief Register the caller's [max_tokens, top_k] fp32 norms buffer.
  ///
  /// Bound once at startup and written every forward. Passing 0 detaches.
  void set_reap_norms_buffer(intptr_t buffer, int max_tokens, int top_k) {
    reap_norms_ = reinterpret_cast<float*>(buffer);
    reap_max_tokens_ = max_tokens;
    reap_top_k_ = top_k;
  }

  /// \brief Norm of each expert output this layer produced, per (token, slot).
  ///
  /// A partition holds a slice of the INTERMEDIATE axis, so its row is full
  /// length but a partial value and f = sum over partitions. The sum therefore
  /// has to happen before the norm: squares add across disjoint COORDINATES,
  /// never across contraction-axis partials. Runs after do_numa_job has
  /// returned, which is the first point at which every partition's rows are
  /// complete, and parallelises over (token, slot) rather than inside one
  /// NUMA node's job -- the reads are cross-node either way, and one node's
  /// threads doing all of them would serialise the work.
  void reap_score(int qlen, int k, const int64_t* expert_ids) {
    // Only the AMX backends expose per-expert rows; everywhere else this
    // compiles away. K3 runs MXFP4 on AMX, which does.
    if constexpr (!requires(T& t, const int64_t* ids) { t.reap_row(0, 0, 0, ids); }) {
      (void)qlen;
      (void)k;
      (void)expert_ids;
      return;
    } else {
    if (reap_norms_ == nullptr || k != reap_top_k_) return;
    // Sized for decode. A longer batch is a prefill shape, and prefill says
    // nothing about residency -- every expert runs on the GPU there -- so
    // refuse it rather than measure a prefix of it.
    if (qlen <= 0 || qlen > reap_max_tokens_) return;
    const int rows = qlen;

    const int hidden = config.hidden_size;
    const int parts = tp_count;
    float* out = reap_norms_;
    auto& parts_ref = tps;
    // A slot this layer does not own reads back zero, which the caller drops.
    std::memset(out, 0, static_cast<size_t>(rows) * k * sizeof(float));

    // Fan out across NUMA nodes. WorkerPool::do_work_stealing_job runs
    // everything on node 0's threads alone, which would leave the reads
    // remote AND serialise them on a sixth of the cores.
    auto* pool = config.pool;
    const int total = rows * k;
    pool->dispense_backend()->do_numa_job([pool, &parts_ref, out, k, hidden, parts,
                                           expert_ids, total](int numa_id) {
      const int begin = static_cast<int>(static_cast<int64_t>(total) * numa_id / parts);
      const int end = static_cast<int>(static_cast<int64_t>(total) * (numa_id + 1) / parts);
      if (end <= begin) return;
      pool->get_subpool(numa_id)->do_work_stealing_job(
        end - begin, nullptr,
        [&parts_ref, out, k, hidden, parts, expert_ids, begin](int slice_t) {
          const int t = begin + slice_t;
          const int i = t / k;
          const int j = t % k;
          const ggml_bf16_t* rows_p[kReapMaxPartitions];
          int n = 0;
          for (int p = 0; p < parts && n < kReapMaxPartitions; ++p) {
            const ggml_bf16_t* r = parts_ref[p]->reap_row(i, j, k, expert_ids);
            if (r != nullptr) rows_p[n++] = r;
          }
          if (n == 0) return;
          double acc = 0.0;
          for (int h = 0; h < hidden; ++h) {
            float v = 0.0f;
            for (int q = 0; q < n; ++q) v += kt_bf16_to_fp32(rows_p[q][h]);
            acc += static_cast<double>(v) * static_cast<double>(v);
          }
          out[i * k + j] = static_cast<float>(std::sqrt(acc));
        },
        nullptr);
    });
    }
  }

  void forward(int* qlen_ptr, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output,
               bool incremental) {
    if (weights_loaded == false) [[unlikely]] {
      throw std::runtime_error("Not Loaded");
    }
#ifdef FORWARD_TIME_REPORT
    auto start = std::chrono::high_resolution_clock::now();
#endif
    int qlen = *qlen_ptr;

    auto pool = config.pool;
    pool->dispense_backend()->do_numa_job([this, pool, qlen, k, expert_ids, input, weights](int numa_id) {
      tps[numa_id]->forward(qlen, k, expert_ids, weights, input, this->local_output_numa[numa_id]);
    });

    reap_score(qlen, k, expert_ids);
    merge_results(qlen, output, incremental);
#ifdef FORWARD_TIME_REPORT
    auto end = std::chrono::high_resolution_clock::now();
    auto forward_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    int unique_experts = 0;
    {
      std::unordered_set<int64_t> expert_set;
      for (int i = 0; i < qlen * config.num_experts_per_tok; i++) {
        expert_set.insert(expert_ids[i]);
      }
      unique_experts = expert_set.size();
    }
    auto band_width =
        (1.0 * unique_experts * config.hidden_size * config.intermediate_size * 3 / 1e9) / (1.0 * forward_time / 1e6);
    auto GFLOPS =
        (1.0 * config.hidden_size * config.intermediate_size * qlen * 3 * config.num_experts_per_tok * 2 / 1e9) /
        (1.0 * forward_time / 1e6);
    if (qlen <= 10) {
      forward_time_sum_ns += forward_time;
      forward_count++;
    }
    auto average_bandwidth =
        (1.0 * forward_count * unique_experts * config.hidden_size * config.intermediate_size * 3 / 1e9) /
        (1.0 * forward_time_sum_ns / 1e6);
    printf(
        "forward time %ld, time stamp:%ld, band width %f GElement/s, ave bandwidth %f GElement/s (only "
        "decode), %f GFLOPS, me numa: %d\n",
        forward_time, end.time_since_epoch().count() / 1000 % 100000000, band_width, average_bandwidth, GFLOPS,
        numa_node_of_cpu(sched_getcpu()));
#endif
  }

  virtual void load_weights() = 0;

  virtual void merge_results(int qlen, void* output) = 0;

  virtual void merge_results(int qlen, void* output, bool incremental) {
    if (incremental == false) {
      merge_results(qlen, output);
    } else {
      throw std::runtime_error("Not Implemented");
    }
  };
};

template <MOE_TP_PART T>
class TP_MOE : public TP_MOE_Common<T> {
 public:
  using TP_MOE_Common<T>::TP_MOE_Common;
  void load_weights(const uint64_t* physical_to_logical_map) { throw std::runtime_error("Not Implemented"); }
  // void merge_results(int qlen, void *output, bool incremental) { throw std::runtime_error("Not Implemented"); }
};

#endif
