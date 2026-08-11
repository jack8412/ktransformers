/**
 * One process-wide lock over the CPU backend's shared job machinery.
 *
 * WHY THIS EXISTS. NumaJobDistributor::do_numa_job publishes work by writing
 * a SHARED member -- `this->compute_func = compute_func;` -- with no lock,
 * then flipping every per-NUMA status to WORKING and spinning until each
 * returns to WAITING (cpu_backend/worker_pool.cpp). InNumaPool's
 * work-stealing entry point has the same shape. There is exactly one
 * WorkerPool per process (CPUInfer is a class-level singleton shared by all
 * 92 layers), so that state is shared by every caller.
 *
 * Nothing in the distributor makes it safe for two callers at once. What made
 * it safe was that EVERY kt entry point -- forwards, load_weights, warm_up,
 * write_weight_scale_to_buffer -- was enqueued onto a single TaskQueue and run
 * FIFO by that queue's one worker thread. The serialization was a property of
 * the caller, not of the distributor.
 *
 * The doorbell transport breaks that premise: it runs the expert forward
 * inline on the POLLER thread (deliberately -- the whole point is to remove a
 * cross-thread hop) while every other entry point stays on the TaskQueue. Two
 * threads can then be inside do_numa_job at once, and the failures are of the
 * worst kind:
 *   - `compute_func` is a std::function; copy-assigning it frees the old
 *     target while a distributor worker may be invoking it -> use-after-free.
 *   - Caller A flips the statuses, caller B overwrites compute_func before
 *     worker i reads it; worker i runs B's job, stores WAITING, and BOTH
 *     spin loops exit satisfied. A's job never ran on that NUMA node, so its
 *     per-socket partial output is left over from the previous step and gets
 *     merged in anyway -- wrong logits, no exception, no counter moved.
 *
 * So: restore the invariant the TaskQueue used to provide, at task
 * granularity rather than at do_numa_job granularity. Locking individual
 * do_numa_job calls would leave a composite operation (a forward is several
 * dispatches plus a merge) interleaved with a weight rewrite, which is
 * precisely what FIFO used to prevent.
 *
 * NESTING: taken only at top-level entry points -- the TaskQueue worker
 * around a whole task body, and the doorbell poller around a whole forward.
 * Distributor/pool worker threads never take it, and no task body enqueues
 * and waits for another, so there is no cycle and a plain mutex suffices.
 *
 * COST: one uncontended lock per task, tens of nanoseconds against a
 * ~10 us transport round trip and ~100 us of expert math.
 */
#ifndef CPUINFER_BACKEND_LOCK_H
#define CPUINFER_BACKEND_LOCK_H

#include <mutex>

inline std::mutex& kt_backend_mutex() {
  static std::mutex m;
  return m;
}

#endif  // CPUINFER_BACKEND_LOCK_H
