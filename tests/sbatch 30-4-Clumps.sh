#!/bin/bash -l
#SBATCH --job-name=ClumpPlotting
#SBATCH --account=rrg-sellison  # ARC allocation; account name def-jwoo/rrg-jfncc
#SBATCH --array=0-22
#SBATCH --time=03:29:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=60960  # in MB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mgp3@sfu.ca
#SBATCH --output=/home/malcolmp/scratch/outputs/output/output_log_%A_%a
#SBATCH --error=/home/malcolmp/scratch/outputs/error/error_log_%A_%a
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK # not really needed as we aren't parallelizing in code
#export MKL_NUM_THREADS=1 # Not really necessary but makes things faster when parallelizing
IDlist=(0 1 2 5826 5827 5829 5832 9000 9002 11343 11459 11533 11745 11746 11881 11931 12048 12049 12660 12863 13728 15152 17137)
taskID=$SLURM_ARRAY_TASK_ID
ID=${IDlist[$taskID]}
Z=3
echo $ID
Snapshot=30
DirSnap='030'
DensThresh='08'
Limit=1000
Dir='ZF4_newSF_fromz127'
source ~/astroEnv/bin/activate
python /home/malcolmp/Notebooks/IPM_Clump_Finding_V4.py $ID $Snapshot $Dir $Limit # In this script, you can read the array ID as SLURM_ARRAY_TASK_ID (see script)
