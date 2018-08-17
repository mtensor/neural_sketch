#Sketch project


from builtins import super
import pickle
import string
import argparse
import random

import torch
from torch import nn, optim

from pinn import RobustFill
import random
import math

from collections import OrderedDict
from util import enumerate_reg, Hole


import sys
sys.path.append("/om/user/mnye/ec")

from grammar import Grammar
from deepcoderPrimitives import deepcoderProductions, flatten_program

import math
from type import tpregex, Context


productions = deepcoderProductions() #TODO - figure out good production probs


grammar = Grammar.fromProductions(productions)

def parseprogram(): #TODO
    #figure out if you want to seperate the fn's and non-fn's
    pass

def sampleIO(): #TODO
    pass

def deepcoder_vocab(n_inputs=3): 
    return grammar.names + ['input_' + str(i) for i in range(n_inputs)] #TODO

def make_holey_deepcoder(prog, k, g):
    choices = g.enumerateHoles(self, request, prog, distance=100.0, k=k)
    progs, weights = zip(*choices)
    if k > 1:
        return random.choices(progs, weights=weights, k=1)[0]
    else:
        return progs[0] #i think it will output a list? #TODO

def sample_request():
    pass

# implement grammar.sampleprogram() #not hard, just grammar.sample should be fine 

    # holey = make_holey_sup_inner(r)
    # return holey, torch.Tensor([scores])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrain_holes', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--start_with_holes', action='store_true')
    #parser.add_argument('--variance_reduction', action='store_true')
    parser.add_argument('--k', type=int, default=1000) #TODO
    args = parser.parse_args()

    max_length = 40
    batch_size = 100


    print("Loading model", flush=True)
    try:
        if args.start_with_holes:
            model=torch.load("./deepcoder_pretrain_holes.p")
            print('found saved model, loading pretrained model with holes')
        else:
            model=torch.load("./deepcoder_pretrained.p")
            print('found saved model, loading pretrained model (without holes)')

    except FileNotFoundError:
        print("no saved model, creating new one")
        model = RobustFill(input_vocabularies=[], target_vocabulary=deepcoder_vocab, max_length=max_length) #TODO

    model.cuda()
    print("number of parameters is", sum(p.numel() for p in model.parameters() if p.requires_grad))

    ######## Pretraining without holes ########

    def getInstance(k_shot=4):
        """
        Returns a single problem instance, as input/target strings
        """
        #TODO
        while True:
            #request = arrow(tlist(tint), tint, tint)
            request = sample_request()
            p = grammar.sample(request) #grammar not abstracted well in this script
            pseq = flatten_program(p)
            IO = [sampleIO(p) for i in range(k_shot)] #TODO
            IO = [IO] #(idk what the right representation is)
            if all(len(x)<max_length for x in IO): break
        return {'IO':IO, 'pseq':pseq, 'p':p}

    def getBatch():
        """
        Create a batch of problem instances, as tensors
        """
        k_shot = random.choice(range(3,6)) #this means from 3 to 5 examples

        instances = [getInstance(k_shot=k_shot) for i in range(batch_size)]
        IO = [inst['IO'] for inst in instances]
        p = [inst['p'] for inst in instances]
        pseq = [inst['pseq'] for inst in instances]
        return IO, pseq, p

    if args.pretrain:

        print("pretraining", flush=True)
        if not hasattr(model, 'pretrain_iteration'):
            model.pretrain_iteration = 0
            model.pretrain_scores = []

        if args.debug:
            Dc, c, _, _, _ = getBatch()

        for i in range(model.pretrain_iteration, 20000):
            if not args.debug:
                IO, pseq, _ = getBatch()

            score = model.optimiser_step(IO, pseq) #TODO make sure inputs are correctly formatted

            model.pretrain_scores.append(score)
            model.pretrain_iteration += 1
            if i%10==0: print("pretrain iteration", i, "score:", score, flush=True)

            #to prevent overwriting model:
            if not args.nosave:
                if i%500==0: torch.save(model, './deepcoder.p')
    ######## End Pretraining without holes ########


    max_length = 30
    batch_size = 100

    ####### train with holes ########
    print("training with holes")
    #model = model.with_target_vocabulary(deepcoder_vocab) #TODO
    model.cuda()

    # if args.variance_reduction:
    #     if not hasattr(model, 'variance_red'):
    #         model.variance_red = nn.Parameter(torch.Tensor([1])).cuda()

    if not args.pretrain_holes:
        optimizer = optim.Adam(model.parameters(), lr=1e-2) #TODO, deal with this

    if not hasattr(model, 'iteration') or args.start_with_holes:
        model.iteration = 0
    if not hasattr(model, 'hole_scores'):
        model.hole_scores = []
    for i in range(model.iteration, 10000):

        if not args.pretrain_holes:
            optimizer.zero_grad()
        IO, pseq, p = getBatch()


        holey_p = [make_holey_deepcoder(prog, args.k) for prog in p] #TODO

        sketch = [flatten_program(prog) for prog in holey_p]

        objective, _ = model.optimiser_step(IO, sketch)

        #TODO: also can try RL objective, but unclear why that would be good.


        model.iteration += 1
        model.hole_scores.append(objective)
        if i%1==0: 
            print("iteration", i, "score:", objective, flush=True)
        if i%100==0:
            inst = getInstance()
            samples, scores, _ = model.sampleAndScore([inst['IO']], nRepeats=100) #Depending on the model
            index = scores.index(max(scores))
            #print(samples[index])
            try: sample = parseprogram(list(samples[index])) #TODO
            except: sample = samples[index]
            #sample = samples[index]
            print("actual program:" )
            print("generated examples:")
            print(*inst['IO'])
            print("inferred:", sample)

        if i%100==0: # and not i==0: 
            if not args.nosave:
                torch.save(model, './deepcoder_holes_ep_{}.p'.format(str(i)))
                torch.save(model, './deepcoder_holes.p')

    ####### End train with holes ########







