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
import time

from collections import OrderedDict
#from util import enumerate_reg, Hole

import sys
sys.path.append("/om/user/mnye/ec")

from grammar import Grammar, NoCandidates
from deepcoderPrimitives import deepcoderProductions, flatten_program
from program import Application, Hole, Primitive, Index, Abstraction, ParseFailure
import math
from type import Context, arrow, tint, tlist, tbool, UnificationFailure
from deepcoder_util import parseprogram, grammar
from makeDeepcoderData import batchloader
from itertools import chain

def deepcoder_vocab(grammar, n_inputs=3): 
    return [prim.name for prim in grammar.primitives] + ['input_' + str(i) for i in range(n_inputs)] + ['<HOLE>'] #TODO

def tokenize_for_robustfill(IOs):
    """
    tokenizes a batch of IOs
    """
    newIOs = []
    for examples in IOs:
        tokenized = []
        for xs, y in examples:
            if isinstance(y, list):
                y = ["LIST_START"] + y + ["LIST_END"]
            else:
                y = [y]
            serializedInputs = []
            for x in xs:
                if isinstance(x, list):
                    x = ["LIST_START"] + x + ["LIST_END"]
                else:
                    x = [x]
                serializedInputs.extend(x)
            tokenized.append((serializedInputs, y))
        newIOs.append(tokenized)
    return newIOs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--start_with_holes', action='store_true')
    #parser.add_argument('--variance_reduction', action='store_true')
    parser.add_argument('-k', type=int, default=3) #TODO
    args = parser.parse_args()

    max_length = 30
    batchsize = 200

    Vrange = 128

    train_data1 = 'data/DeepCoder_data/T2_A2_V512_L10_train.txt'
    loader1 = batchloader(train_data1, batchsize=batchsize, N=5, V=Vrange, L=10, compute_sketches=False)
    train_data2 = 'data/DeepCoder_data/T3_A2_V512_L10_train_perm.txt'
    loader2 = batchloader(train_data2, batchsize=batchsize, N=5, V=Vrange, L=10, compute_sketches=False)
    loader = chain(loader1, loader2)

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
        model = SyntaxCheckingRobustFill(
            input_vocabularies=[list(range(-Vrange, Vrange+1)) + ["LIST_START", "LIST_END"],
            list(range(-Vrange, Vrange+1))+["LIST_START", "LIST_END"]], 
            target_vocabulary=vocab, max_length=max_length) #TODO

    model.cuda()
    print("number of parameters is", sum(p.numel() for p in model.parameters() if p.requires_grad))

    ######## Pretraining without holes ########

    #make this a function...
    if args.pretrain:
        t2 = time.time()

        print("pretraining", flush=True)
        if not hasattr(model, 'pretrain_iteration'):
            model.pretrain_iteration = 0
            model.pretrain_scores = []
        if not hasattr(model, 'pretrain_epochs'):
            model.pretrain_epochs = 0


        for j in range(model.pretrain_epochs, 50): #TODO
            print(f"\tepoch {j}:")

            for i, batch in enumerate(loader):

                IOs = tokenize_for_robustfill(batch.IOs)

                t = time.time()
                score, syntax_score = model.optimiser_step(IOs, batch.pseqs)
                print(f"network time: {time.time()-t}, other time: {t-t2}")
                t2 = time.time()

                model.pretrain_scores.append(score)
                model.pretrain_iteration += 1
                if i%1==0: print("pretrain iteration", i, "score:", score, "syntax score:", syntax_score, flush=True)
                if i%200==0: 
                    if not args.nosave:
                        torch.save(model, './deepcoder_pretrained.p')
        #to prevent overwriting model:
            if not args.nosave:
                torch.save(model, './deepcoder_pretrained.p')
    ######## End Pretraining without holes ########



    ####### train with holes ########

    #make this a function (or class, whatever)
    print("training with holes")
    model = model.with_target_vocabulary(vocab) #TODO
    model.cuda()  # just in case

    if not hasattr(model, 'iteration') or args.start_with_holes:
        model.iteration = 0
    if not hasattr(model, 'hole_scores'):
        model.hole_scores = []
    if not hasattr(model, 'epochs'):
        model.epochs = 0

    t2 = time.time()
    for j in range(model.epochs, 20): #TODO
        for i, batch in enumerate(batchloader(train_data, batchsize=batchsize, N=5, V=Vrange, L=10, compute_sketches=True)):
            IOs = tokenize_for_robustfill(batch.IOs)
            t = time.time()
            objective, syntax_score = model.optimiser_step(IOs, batch.sketchseqs)
            print(f"network time: {time.time()-t}, other time: {t-t2}")
            t2 = time.time()            
            #TODO: also can try RL objective, but unclear why that would be better.
            model.iteration += 1
            model.hole_scores.append(objective)
            if i%1==0: 
                print("iteration", i, "score:", objective, "syntax_score:", syntax_score, flush=True)
            if i%200==0: 
                if not args.nosave:
                    torch.save(model, f'./deepcoder_holes_ep_{str(j)}_iter_{str(i)}.p')
                    torch.save(model, './deepcoder_holes.p')
            
        if not args.nosave:
            torch.save(model, './deepcoder_holes_ep_{}.p'.format(str(j)))
            torch.save(model, './deepcoder_holes.p')


        ####### End train with holes ########







