#!/bin/bash -l
#SBATCH --job-name="prepare_wikipedia"
#SBATCH --nodes=1                  # number of nodes
#SBATCH --ntasks-per-node=1        # Do not change
#SBATCH --gpus-per-node=4          # number of gpus per node
#SBATCH --partition=normal
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00            # total run time limit (HH:MM:SS)

srun --mpi=pmi2 --environment=msamp bash -c "
cd /capstor/scratch/cscs/ctianche/playground/low-precision-testbed/msamp_reprod/msamp_examples/gpt3
bash prepare_wikipedia.sh
"
