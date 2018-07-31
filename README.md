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
- [ ] test first past at make_holey_supervised
- [ ] have some sort of enumeration
- [ ] slightly dynamic batch bullshit so that i don't only take the top 1 sketch, and softmax them together as luke suggested
- [ ] get rid of index in line where sketches are created
- [ ] make scores be pytorch tensor
- [ ] perhaps convert to EC domain for enumeration?? - actually not too hard ... 


