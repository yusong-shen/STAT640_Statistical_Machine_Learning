#!/bin/bash
#SBATCH --job-name=TestRun
#SBATCH --partition=serial
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000m
#SBATCH --time=00:30:00


echo "My job ran on:"
echo $SLURM_NODELIST
# srun is Davinci command which will run the commands which follow on a compute node
# Rscript is an R command which runs script and prints to screen
# BasicRScript.R is the script which Rscript will run
# '> outfile' captures output of Rscript and prints to the file called outfile (instead of printing to screen)
srun Rscript BasicRScript.R > outfile
