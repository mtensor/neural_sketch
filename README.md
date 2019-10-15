# NEURAL SKETCH PROJECT

## Basic idea:
A user should only have to go into the `train` folder, the `eval` folder, and the `plot` folder.
`train` and `eval` folders have train and eval scripts for each domain.
Currently only robustfill and deepcoder are working.


the `train` folder is where the training scripts are. You should run from the top level directory, with the `--pretrain` flag, the first time you run. ex:
```
anaconda-project run python main_supervised_deepcoder.py --pretrain
```
On the MIT openmind computer cluster, the `*.sh` files are used to schedule jobs. I usually do the following:
```
sbatch execute_gpu.sh python main_supervised_deepcoder.py --pretrain
```
(no `anaconda-project` needed with `execute_gpu.sh`)



## THINGS TO NOTE
- for various reasons, the `ec` subdir had to be added to path, so if you are looking at an import statement and don't see the folder in the top level, it's inside `ec/`
- the naming convention around "deepcoder" and "robustfill" is not great. "
- I use the `working-mnye` branch, so it is more up to date, with all submodules, etc.
