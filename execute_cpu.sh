#!/bin/sh

#SBATCH --qos=tenenbaum
#SBATCH --time=3000
#SBATCH --mem=50G
#SBATCH --job-name=neural_sketch_eval
#SBATCH --cpus-per-task=48


#export PATH=/om/user/mnye/miniconda3/bin/:$PATH
#source activate /om/user/mnye/vhe/envs/default/
#cd /om/user/mnye/vhe
anaconda-project run $@
