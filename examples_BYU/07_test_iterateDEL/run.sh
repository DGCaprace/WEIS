#!/bin/bash

#SBATCH --time=04:00:00   # walltime
#SBATCH --ntasks=28   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --cpus-per-task=1
# SBATCH --exclude=m9-57-[1-4]
#SBATCH --partition=m9   # request MaryLou
#SBATCH -C 'avx2'   # features syntax (use quotes): -C 'a&b&c&d'
#SBATCH --mem-per-cpu=2500M   # memory per CPU core
#SBATCH -J "pCrunch"   # job name
# SBATCH --qos=test


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
source /fslhome/dcaprace/ATLANTIS/loadModule_weis.sh

# SCRATCH=/lustre/scratch/usr/dcaprace/2021_UAV_gcc/DJI3_fwd10

# refdir=$PWD
 #outdir=$SCRATCH/fine_4fwd10_p13_a0_rpm4000_$SLURM_JOB_ID/
 
# mkdir -p $outdir

# cp $refdir/* $outdir

echo " "
echo "Job $SLURM_JOBID"
echo "Nodelist : $SLURM_NODELIST"
echo "a = NTASKS               = $SLURM_NTASKS"
echo "b = SLURM_CPUS_PER_TASK  = $SLURM_CPUS_PER_TASK"
echo "c = SLURM_TASKS_PER_NODE = $SLURM_TASKS_PER_NODE"
echo " "
echo "Starting job in $outdir"
echo " "

# cd $outdir


mpirun -np $SLURM_NTASKS -tag-output python driver.py > stdout_${SLURM_JOB_ID}.out


echo "DONE"

scontrol show job $SLURM_JOB_ID
