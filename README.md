# hmd-cuda-graph

**CSCI 596 (Scientific Computing and Visualization) Final Project — Programming track**

Triple-decker **MPI + OpenMP + CUDA** molecular dynamics, extending the hybrid
MPI+OpenMP `hmd.c` from Assignment 5 by porting the force-computation kernel,
`compute_accel()`, to the GPU — and then optimizing the per-timestep kernel
sequence with **CUDA Graphs** to eliminate repeated launch overhead.

## Goal

`compute_accel()` is called twice per MD timestep. For modest system sizes,
each of the handful of kernels involved (build cell list, pair-force kernel,
half-kicks, …) runs in only a few microseconds on the GPU — comparable to or
smaller than the CPU-side launch overhead of `cudaLaunchKernel`. Across the
hundreds of thousands of timesteps a typical MD run takes, that overhead adds
up and the GPU sits idle waiting on the CPU to issue the next kernel.

This project ports the force kernel to CUDA, confirms correctness against the
CPU reference, then captures the fixed per-step kernel sequence into a
**CUDA Graph** and replays it instead of re-issuing individual launches —
benchmarking the launch-overhead-bound regime (small systems) against the
compute-bound regime (large systems) to show where Graphs pay off.

## Architecture (3 layers, same spatial decomposition as `hmd.c`)

```
MPI (inter-node, domain decomposition)        -- unchanged from hmd.c
 └─ OpenMP (one thread per GPU on the node)    -- unchanged pattern from pi3.cu
     └─ CUDA (per-thread cell-block force kernel, on its own cudaSetDevice)
```

- **MPI layer**: unchanged from `hmd.c` — `atom_copy()` / `atom_move()` still
  run on the host, exchanging boundary atoms between ranks.
- **OpenMP layer**: each MPI rank spawns `NUM_DEVICE` threads; each thread
  owns one GPU (`cudaSetDevice(mpid % NUM_DEVICE)`) and the sub-block of
  linked-list cells that thread owned in `hmd.c`'s CPU version.
- **CUDA layer**: the pair-force double loop over a cell block, ported from
  `compute_accel()`'s inner loop (same linked-cell / minimum-image logic as
  `pdf.cu`'s GPU histogram kernel).

## Plan / milestones

- [x] **Phase 0** — repo skeleton, CPU baseline (`hmd_cpu.c`, copied from
      Assignment 5) builds and runs as a correctness reference.
- [x] **Phase 1** — single-GPU, single-MPI-rank CUDA port of
      `compute_accel()` (`src/hmd_gpu.cu`). Validated on CARC (`gpu` partition,
      job 13733768): total energy matches the CPU reference exactly to every
      printed digit across all 10 timesteps, and the GPU kernel runs ~9.5x
      faster on compute time (0.68s vs 6.51s for `CPU & COMT`).
- [ ] **Phase 2** — wrap into the full MPI + OpenMP(1 thread/GPU) + CUDA
      structure, matching `pi3.cu`'s multi-GPU pattern.
- [ ] **Phase 3** — capture the per-step kernel sequence as a CUDA Graph
      (`cudaGraphExec`), replayed instead of re-launched each step; update
      graph node parameters between steps instead of re-capturing.
- [ ] **Benchmark** — launch-overhead-bound (small N) vs compute-bound
      (large N) comparison, naive-GPU vs Graph-optimized, plotted.

## Build

```
make            # builds the CPU reference binary, hmd_cpu
make gpu        # (Phase 1+) builds the CUDA binary, hmd_gpu
```

## Layout

```
include/hmd.h        Shared header (from Assignment 5)
src/hmd_cpu.c         CPU reference (= Assignment 5's hmd.c, for validation)
src/hmd_gpu.cu         CUDA port of compute_accel()            [Phase 1]
src/hmd_graph.cu          + CUDA Graph optimization             [Phase 3]
scripts/               Slurm scripts for CARC Discovery
benchmarks/            Timing data & plots
pmd.in                 MD run parameters (from Assignment 5)
```

## Status

Phase 1 validated. Starting Phase 2 (multi-GPU MPI + OpenMP + CUDA).
