# Neural Sketch project
- learning to learn sketches for programs

#TODO:
- [ ] Get regexes working 
- [ ] figure out correct objective for learning holes -completed ish - no
- [ ] start by training holes on pretrain
- [ ] have lower likelihood of holes
- [ ] make hole probibility vary with depth
- [ ] fuss with correct way to generate sketches
- [ ] Use more sophisticated nn architectures (perhaps only after getting RobustFill to work)
- [ ] Try other domains??


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
- [ ] primitive set for deepcoder
- [ ] implement the program.flatten() method for nns (super easy)
- [ ] implement parsing from string (not bad either, use stack machine )
- [ ] implement training (pretrain and makeholey)
- [ ] implement evaluation
- [ ] test out reversing nn implementation


#TODO for refactoring: 
- [ ] make one main (domain agnostic) training script/file
- [ ] make one main (domain agnostic) evaluation script/file


#TODO for TPUs:
- [ ] Graph networks?
- [ ] TPU stuff
- [ ] read brian's stuff (https://github.com/tensorflow/minigo/blob/master/dual_net.py)

