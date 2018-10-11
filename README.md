# NEURAL SKETCH PROJECT

## Basic idea:
A user should only have to go into the `train` folder, the `eval` folder, and the `plot` folder.
`train` and `eval` folders have train and eval scripts for each domain.
Currently only robustfill and deepcoder are working.


the `train` folder is where the training scripts are. You should run from the top level directory, with the `--pretrain` flag, the first time you run. ex:
```
anaconda-project run python main_supervised_deepcoder.py --pretrain
```
On openmind, I usually do the following:
```
sbatch execute_gpu.sh python main_supervised_deepcoder.py --pretrain
```
(no `anaconda-project` needed with `execute_gpu.sh`)



## THINGS TO NOTE
- for various reasons, the `ec` subdir had to be added to path, so if you are looking at an import statement and don't see the folder in the top level, it's inside `ec/`
- the util directory is a mess. I will clean it
- the naming convention around "deepcoder" and "robustfill" is bad ... sorry ...

## OCTOBER CLEAN UP
- [X] switch to hierarchical file structure
- [X] add EC as submodule or something
- [ ] fix 'alternate' bug in evaluate code
	- [ ] eval script
	- [ ] loader scripts?
- [ ] possibly find better names for saved things
- [ ] remove all magic values
- [ ] deal with silly sh scripts
- [ ] fix readme for other users
- [ ] run those other tests
- [ ] perhaps redesign results stuff
- [ ] make sure pypy stuff still works
- [ ] make sure saved models work
- [ ] figure out what needs to be abstracted, and abstract
- [ ] run dc with smaller train 4 split

folders to comb through for hierarchical struct:
- [X] train
- [X] eval
- [X] tests
- [X] data_src
- [X] models
- [X] plot
- [X] utils
