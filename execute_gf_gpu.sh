#!/bin/sh

#SBATCH --qos=tenenbaum
#SBATCH --time=2000
#SBATCH --mem=30G
#SBATCH --job-name=neural_sketch
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1


#export PATH=/om/user/mnye/miniconda3/bin/:$PATH
#source activate /om/user/mnye/vhe/envs/default/
#cd /om/user/mnye/vhe
anaconda-project run $@
