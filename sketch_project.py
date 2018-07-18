#Sketch project


from builtins import super
import pickle
import string
import argparse
import random

import torch
from torch import nn, optim

from pinn import RobustFill
import pregex as pre
from vhe import VHE, DataLoader, Factors, Result, RegexPrior
import random
import math


regex_prior = RegexPrior()
#k_shot = 4


class Hole(pre.Pregex):
    def __new__(cls): return super(Hole, cls).__new__(cls, None)
    def __repr__(self): return "(HOLE)"
    def flatten(self, char_map={}, escape_strings=False):
        return [type(self)]
    def walk(self, depth=0):
        """
        walks through the nodes
        """
        yield self, depth



regex_vocab = list(string.printable[:-4]) + \
    [pre.OPEN, pre.CLOSE, pre.String, pre.Concat, pre.Alt, pre.KleeneStar, pre.Plus, pre.Maybe, Hole] + \
    regex_prior.character_classes


def make_holey(r: pre.Pregex, p=0.2) -> (pre.Pregex, torch.Tensor):
    """
    makes a regex holey
    """
    scores = 0
    def make_holey_inner(r: pre.Pregex) -> pre.Pregex:
        if random.random() < p: 
            nonlocal scores
            scores += regex_prior.scoreregex(r)
            return Hole()
        else: 
            return r.map(make_holey_inner)

    holey = make_holey_inner(r)
    return holey, torch.Tensor([scores])

def sketch_logprior(preg: pre.Pregex, p=0.2) -> torch.Tensor:
    logprior=0
    for r, d in preg.walk():
        if type(r) is pre.String or type(r) is pre.CharacterClass or type(r) is Hole: #TODO, is a leaf
            if type(r) is Hole: #TODO
                logprior += math.log(p) + d*math.log(1-p)
            else:
                logprior += (d+1)*math.log(1-p)

    return torch.tensor([logprior])




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrain_holes', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--start_with_holes', action='store_true')
    args = parser.parse_args()

    max_length = 30
    batch_size = 200

    print("Loading model", flush=True)
    try:
        if args.start_with_holes:
            model=torch.load("./sketch_model_pretrain_holes.p")
            print('found saved model, loading pretrained model with holes')
        else:
            model=torch.load("./sketch_model_pretrained.p")
            print('found saved model, loading pretrained model')

    except FileNotFoundError:
        print("no saved model, creating new one")
        model = RobustFill(input_vocabularies=[string.printable[:-4]], target_vocabulary=regex_vocab, max_length=max_length)

    model.cuda()
    print("number of parameters is", sum(p.numel() for p in model.parameters() if p.requires_grad))


    ######## Pretraining without holes ########

    def getInstance(k_shot=4):
        """
        Returns a single problem instance, as input/target strings
        """
        #k_shot = 4 #random.choice(range(3,6)) #this means from 3 to 5 examples
 
        while True:
            r = regex_prior.sampleregex()
            c = r.flatten()
            x = r.sample()
            Dc = [r.sample() for i in range(k_shot)]
            c_input = [c]
            if all(len(x)<max_length for x in Dc + [c, x]): break
        return {'Dc':Dc, 'c':c, 'c_input':c_input, 'x':x, 'r':r}

    def getBatch():
        """
        Create a batch of problem instances, as tensors
        """
        k_shot = random.choice(range(3,6)) #this means from 3 to 5 examples

        instances = [getInstance(k_shot=k_shot) for i in range(batch_size)]
        Dc = [inst['Dc'] for inst in instances]
        c = [inst['c'] for inst in instances]
        c_input = [inst['c_input'] for inst in instances]
        x = [inst['x'] for inst in instances]
        r = [inst['r'] for inst in instances]
        return Dc, c, c_input, x, r 

    if args.pretrain:

        print("pretraining", flush=True)
        if not hasattr(model, 'pretrain_iteration'):
            model.pretrain_iteration = 0
            model.pretrain_scores = []

        if args.debug:
            Dc, c, _, _, _ = getBatch()

        for i in range(model.pretrain_iteration, 20000):
            if not args.debug:
                Dc, c, _, _, _ = getBatch()
            #print("Dc:", Dc)
            #print("c:", c)
            score = model.optimiser_step(Dc, c)

            model.pretrain_scores.append(score)
            model.pretrain_iteration += 1
            if i%10==0: print("pretrain iteration", i, "score:", score, flush=True)

            #to prevent overwriting model:
            if not args.nosave:
                if i%500==0: torch.save(model, './sketch_model.p')
    ######## End Pretraining without holes ########


    max_length = 30
    batch_size = 100

    ####### train with holes ########
    print("training with holes")
    model = model.with_target_vocabulary(regex_vocab)
    model.cuda()

    if not args.pretrain_holes:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if not hasattr(model, 'iteration') or args.start_with_holes:
        model.iteration = 0
    if not hasattr(model, 'hole_scores'):
        model.hole_scores = []
    for i in range(model.iteration, 10000):

        if not args.pretrain_holes:
            optimizer.zero_grad()
        Dc, c, _, _, r = getBatch()

        holey_r, holescore = zip(*[make_holey(reg) for reg in r])
        sketch = [reg.flatten() for reg in holey_r]

        if not args.pretrain_holes:
            holescore = torch.cat(holescore, 0).cuda()
            full_program_score = model.score(Dc, c, autograd=False)

            #sketch_prior = torch.cat(tuple(sketch_logprior(sk) for sk in holey_r), 0).cuda()

            #put holes into r
            #calculate score of hole

            #I think this is the right objective ... check https://arxiv.org/pdf/1506.05254.pdf - actually don't, not relevant

            #print(holescore.size())
            #print(full_program_score.size()
            #print(full_program_score)
            #print(torch.exp(-full_program_score))
            #objective = model.score(Dc, sketch, autograd=True)*torch.exp(holescore)*torch.exp(-full_program_score)
            objective = model.score(Dc, sketch, autograd=True)*torch.exp(holescore)
            #objective = model.score(Dc, sketch, autograd=True)*holescore

            """log E_{S~Q) P(y|S)
>= E_{S~Q) log P(y|S)
= E{S~R} Q(S)/R(S) log P(y|S)"""

            #objective = (torch.exp(model.score(Dc, sketch, autograd=True)) / torch.exp(sketch_prior) * holescore)
            #objective = model.score(Dc, sketch, autograd=True) / torch.exp(sketch_prior) * holescore

            #objective = model.score(Dc, sketch, autograd=True)*(holescore - sketch_prior)
            #control:
            #objective = model.score(Dc, c, autograd=True)
            #control 2:
            #objective = model.score(Dc, sketch, autograd=True)

            #objective = model.score(Dc, sketch, autograd=True)*(holescore - full_program_score)
            #print(objective.size())
            objective = objective.mean()
            #print(objective)
            (-objective).backward()
            optimizer.step()

        else: #if args.pretrain_holes:
            objective = model.optimiser_step(Dc, sketch)

        model.iteration += 1
        model.hole_scores.append(objective)
        if i%10==0: 
            print("iteration", i, "score:", objective, flush=True)
        if i%100==0:
            inst = getInstance()
            samples, scores = model.sampleAndScore([inst['Dc']], nRepeats=100)
            index = scores.index(max(scores))
            #print(samples[index])
            try: sample = pre.create(list(samples[index]))
            except: sample = samples[index]
            sample = samples[index]
            print("actual program:", pre.create(inst['c']))
            print("generated examples:")
            print(*inst['Dc'])
            print("inferred:", sample)

        if i%500==0: # and not i==0: 
            if not args.nosave:
                torch.save(model, './sketch_model_holes.p')

    ####### End train with holes ########



    ######testing######


    # ###### full RL training with enumeration ########
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # for batch in actual_data: #or synthetic data, whatever:
    #     optimizer.zero_grad()
    #     samples, scores = model.sampleAndScore(batch, autograd=True) #TODO: change so you can get grad through here, and so that scores are seperate???
    #     objective = []
    #     for sample, score, examples in zip(samples, scores, batch):
    #         #DO RL
    #         objective.append = -score*find_ll_reward_with_enumeration(sample, examples, time=10) #TODO, ideally should be per hole


    #     objective = torch.sum(objective)
    #     #DO RL 
    #     #TODO???? oh god. Make reward in positive range??) 
    #     #Q: is it even

    #     objective.backward()
    #     optimizer.step()

    #from holes, enumerate:
    #option 1: use ec enumeration
    #option 2: use something else
    """
    RL questions:
    - should I do param updates for each batch???
    - can we even get gradients through the whole sample? no, but not too hard I think
    - 
    """
    #trees will be nice because you can enumerate within the tree by just sampling more --- then is it even worth it??
    ###### End full training ########




    #informal testing:







