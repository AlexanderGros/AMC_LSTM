#!/bin/bash
# Submission script 
#SBATCH --job-name=lstm_st_128_256_128
#SBATCH --time=0-15:15:00 # d-hh:mm:ss  60h
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=12288 # 120GB   36864  #  8192 12288  24576  36864  49152
#SBATCH --partition=batch,long
#
#SBATCH --mail-user=alexander.gros@umons.ac.be
#SBATCH --mail-type=ALL
#SBATCH --output=lstm_st_128_256_128.out



#module purge
#ml SciPy-bundle/2023.11-gfbf-2023b
ml --quiet purge
ml --quiet releases/2023a
ml SciPy-bundle/2023.07-gfbf-2023a
ml Python
ml matplotlib/3.7.2-gfbf-2023a
ml h5py/3.9.0-foss-2023a


#ml --quiet purge
#ml --quiet releases/2023a
#ml Python
#ml matplotlib/3.7.2-gfbf-2023a
#ml h5py/3.9.0-foss-2023a

#ml --quiet purge
#ml --quiet releases/2020b
#ml Keras/2.4.3-foss-2020b
#ml matplotlib/3.3.3-foss-2020b
#ml h5py/3.1.0-foss-2020b



echo "job start at $(date)"
python lstm_standard.py
echo "job end at $(date)"