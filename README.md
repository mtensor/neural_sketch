#NEURAL SKETCH PROJECT

the `train` folder is where the training scripts are. You should run from the top level directory, with the `--pretrain` flag, the first time you run. ex:
```
anaconda-project main_supervised_deepcoder.py --pretrain
```

Basic idea:


#THINGS TO NOTE
- for various reasons, the `ec` subdir had to be added to path, so if you are looking at an import statement and don't see the folder in the top level, it's inside `ec/`
- the util directory is a mess. I will clean it
- the naming convention around "deepcoder" and "robustfill" is bad ... sorry ...

#OCTOBER CLEAN UP
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

folders to comb through for hierarchical struct:
- [X] train
- [X] eval
- [X] tests
- [X] data_src
- [X] models
- [X] plot
- [X] utils

- [ ] run dc with smaller train 4 split