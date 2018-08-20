#Sketch project


from builtins import super
import pickle
import string
import argparse
import random

import torch
from torch import nn, optim

from pinn import RobustFill
from pinn import SyntaxCheckingRobustFill #TODO
import random
import math

from collections import OrderedDict
#from util import enumerate_reg, Hole


import sys
sys.path.append("/om/user/mnye/ec")

from grammar import Grammar
from deepcoderPrimitives import deepcoderProductions, flatten_program

from program import Application, Hole

import math
from type import Context, arrow, tint, tlist, tbool, UnificationFailure



productions = deepcoderProductions() #TODO - figure out good production probs
grammar = Grammar.fromProductions(productions)

def parseprogram(): #TODO 
    pass


def isListFunction(tp):
    try:
        Context().unify(tp, arrow(tlist(tint), tint)) #TODO, idk if this will work
        return True
    except UnificationFailure:
        try:
            Context().unify(tp, arrow(tlist(tint), tlist(tint))) #TODO, idk if this will work
            return True
        except UnificationFailure:
            return False


def isIntFunction(tp):
    try:
        Context().unify(tp, arrow(tint, tint)) #TODO, idk if this will work
        return True
    except UnificationFailure:
        try:
            Context().unify(tp, arrow(tint, tlist(tint))) #TODO, idk if this will work
            return True
        except UnificationFailure:
            return False

def sampleIO(program, tp, k_shot=4): #TODO
    #needs to have constraint stuff
    N_EXAMPLES = 5
    RANGE = 30
    LIST_LEN_RANGE = 8
    #stolen from Luke. Should be abstracted in a class of some sort.
    def _featuresOfProgram(program, tp, k_shot=4):
        e = program.evaluate([])
        examples = []
        if isListFunction(tp):
            sample = lambda: random.sample(range(RANGE), random.randint(0, LIST_LEN_RANGE))
        elif isIntFunction(tp):
            sample = lambda: random.randint(0, RANGE)
        else:
            return None
        for _ in range(N_EXAMPLES*5):
            x = sample()
            try:
                y = e(x)
                #eprint(tp, program, x, y)
                if type(y) == int:
                    y = [y] #TODO fix this dumb hack ...    
                if type(x) == int:
                    x = [x] #TODO fix this dumb hack ...  


                examples.append( (x, y) )


            except: continue
            if len(examples) >= k_shot: break
        else:
            return None #What do I do if I get a None?? Try another program ...
        return examples

    return _featuresOfProgram(program, tp, k_shot=k_shot)

def sample_request(): #TODO
    requests = [
    arrow(tlist(tint), tlist(tint)),
    arrow(tlist(tint), tint),
    #arrow(tint, tlist(tint)),
    arrow(tint, tint)
    ]

    return random.choices(requests, weights=[4,3,1])[0] #TODO

def deepcoder_vocab(grammar, n_inputs=3): 
    return grammar.primitives + ['input_' + str(i) for i in range(n_inputs)] #TODO

def make_holey_deepcoder(prog, k, g, request): #(prog, args.k, grammar, request)
    choices = g.enumerateHoles(request, prog, distance=100.0, k=k) 
    print("choices", list(choices))
    progs, weights = zip(*choices)
    if k > 1:
        return random.choices(progs, weights=weights, k=1)[0]
    else:
        return progs[0] #i think it will output a list? #TODO

def getInstance(k_shot=4):
    """
    Returns a single problem instance, as input/target strings
    """
    #TODO
    #rint("starting getIntance")
    while True:
        #request = arrow(tlist(tint), tint, tint)
        #print("starting getIntance loop")
        request = sample_request()
        #print("request", request)
        p = grammar.sample(request) #grammar not abstracted well in this script
        #print("program:", p)
        pseq = flatten_program(p)
        IO = sampleIO(p, request, k_shot)
        
        if IO == None: #TODO, this is a hack!!!
            continue
        if any(y==None for x,y in IO):
            continue

        #IO = [IO] #(idk what the right representation is)
        #print("IO:", IO)
        if all(len(x)<max_length and len(y)<max_length for x, y in IO): break
    return {'IO':IO, 'pseq':pseq, 'p':p, 'tp': request}

def getBatch():
    """
    Create a batch of problem instances, as tensors
    """
    k_shot = random.choice(range(3,6)) #this means from 3 to 5 examples

    instances = [getInstance(k_shot=k_shot) for i in range(batch_size)]
    IO = [inst['IO'] for inst in instances]
    p = [inst['p'] for inst in instances]
    pseq = [inst['pseq'] for inst in instances]
    tp = [inst['tp'] for inst in instances]
    return IO, pseq, p, tp 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrain_holes', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--start_with_holes', action='store_true')
    #parser.add_argument('--variance_reduction', action='store_true')
    parser.add_argument('--k', type=int, default=3) #TODO
    args = parser.parse_args()

    max_length = 30
    batch_size = 100


    vocab = deepcoder_vocab(grammar)

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
        model = SyntaxCheckingRobustFill(input_vocabularies=[], target_vocabulary=vocab, max_length=max_length) #TODO

    model.cuda()
    print("number of parameters is", sum(p.numel() for p in model.parameters() if p.requires_grad))

    ######## Pretraining without holes ########

    #make this a function...
    if args.pretrain:
        print("pretraining", flush=True)
        if not hasattr(model, 'pretrain_iteration'):
            model.pretrain_iteration = 0
            model.pretrain_scores = []

        if args.debug:
            IO, pseq, _, _ = getBatch()

        for i in range(model.pretrain_iteration, 20000):
            if not args.debug:
                IO, pseq, _, _ = getBatch()

            score, _ = model.optimiser_step(IO, pseq) #TODO make sure inputs are correctly formatted
            model.pretrain_scores.append(score)
            model.pretrain_iteration += 1
            if i%10==0: print("pretrain iteration", i, "score:", score, flush=True)

            #to prevent overwriting model:
            if not args.nosave:
                if i%500==0: torch.save(model, './deepcoder.p')
    ######## End Pretraining without holes ########



    ####### train with holes ########

    #make this a function (or class, whatever)
    print("training with holes")
    #model = model.with_target_vocabulary(deepcoder_vocab) #TODO
    model.cuda()

    if not hasattr(model, 'iteration') or args.start_with_holes:
        model.iteration = 0
    if not hasattr(model, 'hole_scores'):
        model.hole_scores = []
    for i in range(model.iteration, 10000):

        IO, pseq, p, tp = getBatch()
        #assert False
        holey_p = [make_holey_deepcoder(prog, args.k, grammar, request) for prog, request in zip(p, tp)] #TODO
        sketch = [flatten_program(prog) for prog in holey_p]
        objective, _ = model.optimiser_step(IO, sketch)

        #TODO: also can try RL objective, but unclear why that would be better.

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
            print(inst['p'])
            print("generated examples:")
            print(*inst['IO'])
            print("inferred:", sample)

        if i%500==0: # and not i==0: 
            if not args.nosave:
                torch.save(model, './deepcoder_holes_ep_{}.p'.format(str(i)))
                torch.save(model, './deepcoder_holes.p')

    ####### End train with holes ########







