#!/bin/bash
# Benchmark sweep: naive GPU (Phase 2) vs CUDA Graph (Phase 3) across
# system sizes, to find where graph capture/replay actually pays off.
# Run from the repo root (where hmd_gpu / hmd_graph were built):
#   sbatch benchmarks/sweep/run_sweep.sl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:14:59
#SBATCH --output=benchmarks/sweep/sweep.out
#SBATCH -A anakano_429

module purge
module load usc
module load cuda
module load openmpi

cp pmd.in pmd.in.bak

for size in tiny small medium large; do
  echo "##### size=$size #####"
  cp benchmarks/sweep/$size/pmd.in pmd.in
  echo "--- hmd_gpu (Phase 2, naive) ---"
  mpirun -n 2 ./hmd_gpu
  echo "--- hmd_graph (Phase 3, CUDA Graph) ---"
  mpirun -n 2 ./hmd_graph
done

mv pmd.in.bak pmd.in
echo "##### sweep done #####"
