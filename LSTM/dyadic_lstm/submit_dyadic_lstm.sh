#!/bin/bash
# Submission script 
#SBATCH --job-name=dyadic_lstm
#SBATCH --time=2-08:30:00 # d-hh:mm:ss 860h     3-08
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=12288 # 120GB   36864  #  8192 12288  24576  36864  49152
#SBATCH --partition=batch,long
#
# #SBATCH --mail-user=alexander.gros@umons.ac.be
# #SBATCH --mail-type=ALL
#SBATCH --output=dyadic_lstm_2.out



#module purge
ml --quiet purge
ml --quiet releases/2023a
ml SciPy-bundle/2023.07-gfbf-2023a
ml matplotlib/3.7.2-gfbf-2023a
ml Python



echo "job start at $(date)"
python dyadic_lstm.py
echo "job end at $(date)"

