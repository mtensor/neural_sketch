# Neural Sketch project
- learning to learn sketches for programs

#TODO:
- [x] Get regexes working 
- [ ] figure out correct objective for learning holes -completed ish - no
- [x] start by training holes on pretrain
- [x] have lower likelihood of holes
- [x] make hole probibility vary with depth
- [x] fuss with correct way to generate sketches
- [ ] Use more sophisticated nn architectures (perhaps only after getting RobustFill to work)
- [x] Try other domains??


#New TODO:
- [ ] use fully supervised version of holes
- [ ] more sophisiticated nn architectures



#TODO for fully supervised (`at main_supervised.py`):
- [X] implement first pass at make_holey_supervised
- [X] test first pass at make_holey_supervised
- [X] have some sort of enumeration
- [ ] slightly dynamic batch bullshit so that i don't only take the top 1 sketch, and softmax them together as luke suggested
- [ ] get rid of index in line where sketches are created
- [ ] make scores be pytorch tensor
- [ ] perhaps convert to EC domain for enumeration?? - actually not too hard ... 
- [ ] Kevin's continuation representation (see notes)

#TODO for PROBPROG 2018 submission:
- [ ] make figures:
	- [X] One graphic of model for model section
	- [X] One example of data
	- [X] One example of model outputs or something
	- [ ] One results graph, if i can swing it?
	- [X] Use regex figure from nampi submission?
	- [ ] possibly parallelize for evaluation (or not)
	- [ ] write paper
	- [X] show examples and stuff
	- [ ] explain technical process
	- [ ] write intro
	- [ ] remember to show 



#TODO for ICLR submission
- [ ] refactor/generalize evaluation
- [ ] beam search/Best first search
- [ ] multiple holes (using EC langauge)
- [X] build up Syntax-checker LSTM


#TODO for DEEPCODER for ICLR:
- [ ] dataset/how to train
- [X] primitive set for deepcoder
- [X] implement the program.flatten() method for nns (super easy)
- [ ] implement parsing from string (not bad either, use stack machine )
- [X] implement training (pretrain and makeholey)
- [ ] implement evaluation
- [ ] test out reversing nn implementation

#ACTUAL TODO for DEEPCODER ICLR
Training:
- [X] DSL weights (email dude) - x 
- [ ] generating train/test data efficiently
- [ ] offline dataset generation
- [ ] constraint based data gen
- [ ] pypy for data gen
- [X] modify mutation code so no left_application is a hole
- [X] adding HOLE to parser when no left_application is a hole
- [ ] Compare no left_application is a hole to the full case where everything can be a hole
- [ ] write parser for full case 
- [X] add HOLE to nn
- [ ] dealing with different request types
- [ ] multple holes (modify mutator)
- [ ] deepcoder recognition model in the loop
- [ ] simple deepcoder baseline
- [ ] make syntaxrobustfill have same output signature as regular robustfill
- [X] limit depth of programs
- [X] offline dataset collection and training
	- [X] generation
	- [X] training
	- [X] with sketch stuff
- [ ] filtering out some dumb programs (ex lambda $0)
- [ ] use actual deepcoder data, write converter
- [ ] deal with issue of different types and IO effectively in a reasonable manner
- [ ] incorporate constraint based stuff from Marc


Evaluation:
- [X] parsing w/Holes - sorta
- [ ] beam search 
- [ ] figure out correct evaluation scheme 
- [ ] parallelization? (dealing with eval speed)
- [ ] good test set
- [ ] validation set
- [ ] test multiple holes training/no_left_application vs other case
- [ ] using speed

Overall:
- [ ] run training and evaluation (on val set) together for multple experiments to find best model 
- [ ] refactor everything (a model Class, perhaps??)

Tweaking:
- [ ] tweak topk
- [ ] neural network tweaks for correct output format (deal with types and such)
- [ ] 

#TODO for refactoring: 
- [ ] make one main (domain agnostic) training script/file
- [ ] make one main (domain agnostic) evaluation script/file
- [ ] figure out the correct class structure to make work easier to extend and tweak.


#TODO for NAPS:
- [ ] read their code
- [ ] understand how to extend when needed for enumeration stuff
- [ ] make validation set?
- [ ] EMAIL THEM FOR ADDITIONAL QUESTIONS! TRY TO KNOW WHAT TO ASK FOR BY EOD FRIDAY. (IDEAL)
- [ ] make choices such as types of holes, etc.
- [ ] think about how to enumerate, etc. 

#TODO for TPUs:
- [ ] Graph networks?
- [ ] TPU stuff
- [ ] read brian's stuff (https://github.com/tensorflow/minigo/blob/master/dual_net.py)


#FRIDAY TODO:
- [X] loading dataset
- [X] evaluation code, apart from IO concerns







