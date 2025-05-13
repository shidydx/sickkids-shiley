#!/bin/bash
#SBATCH -J shileyJob
#SBATCH -e desired_error_location/%x.oe%j
#SBATCH -o desired_output_location/%x.oe%j
#SBATCH -N 1                  
#SBATCH -c 15                  
#SBATCH -t 0-10:00:00 

module purge
module load miniconda/24.5
eval "$(conda shell.bash hook)"
conda activate #desired_packages_location


cd #file_location
python #python_file