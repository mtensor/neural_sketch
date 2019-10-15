# NEURAL SKETCH PROJECT
This is the code used for the ICML 2019 paper [Learning to Infer Program Sketches](https://arxiv.org/abs/1902.06349).


## Usage:
A user should only have to go into the `train` folder, the `eval` folder, and the `plot` folder.
`train` and `eval` folders have train and eval scripts for each domain.


the `train` folder is where the training scripts are. You should run from the top level directory, with the `--pretrain` flag, the first time you run. ex:
```
anaconda-project run python main_supervised_deepcoder.py --pretrain
```


To fully train the SketchAdapt system, first train the synthesizer (referred to as the `dc_model` in the codebase):
```
python train/deepcoder_train_dc_model.py
```
and pretrain the sketch generator:
```
python train/main_supervised_deepcoder.py --pretrain
```
Then train the sketch generator:
```
python train/main_supervised_deepcoder.py
```
Evaluation can be run with:
```
python eval/evaluate_deepcoder.py
```


NB: On the MIT openmind computer cluster, the `*.sh` files are used to schedule jobs. I usually do the following:
```
sbatch execute_gpu.sh python main_supervised_deepcoder.py --pretrain
```

## THINGS TO NOTE
- for various reasons, the `ec` subdir had to be added to path, so if you are looking at an import statement and don't see the folder in the top level, it's inside `ec/`
- the naming convention around "deepcoder" and "robustfill" is not great. "dc" is often used to represent the 
- I use the `working-mnye` branch, so it is more up to date, with all submodules, etc. If you can't find something on `master`, look it `working-mnye`.
