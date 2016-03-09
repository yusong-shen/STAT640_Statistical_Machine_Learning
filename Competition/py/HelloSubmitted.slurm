#!/bin/bash
#SBATCH --job-name=PyPMBenchmark
#SBATCH --partition=serial
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1000m
#SBATCH --time=00:30:00
#SBATCH --mail-user=yusongtju@gmail.com
#SBATCH --mail-type=ALL


echo "My job ran on:"
echo $SLURM_NODELIST
# srun is Davinci command which will run the commands which follow on a compute node
srun python hello.py >> $HOME/out.txt
