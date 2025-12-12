#!/bin/bash
#SBATCH --partition=edr1-al9_short_serial
#SBATCH --job-name=intro_ML

#SBATCH --ntasks=1           
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G          
#SBATCH --output=/ceph/work/SATORI/alex/intro_ML/out/out_%A_%a.out
#SBATCH --error=/ceph/work/SATORI/alex/intro_ML/err/err_%A_%a.err
#SBATCH --array=0-99

cd /dicos_ui_home/alex/intro_ML_project/tensor_build


source /dicos_ui_home/alex/anaconda3/etc/profile.d/conda.sh
conda activate prometheus_2.0

CHUNK_IDX=${SLURM_ARRAY_TASK_ID}
python build.py --chunk-idx ${CHUNK_IDX} --files-per-chunk 5
