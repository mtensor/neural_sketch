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





def deepcoder_vocab(grammar, n_inputs=3): 
    return [prim.name for prim in grammar.primitives] + ['input_' + str(i) for i in range(n_inputs)] + ['<HOLE>'] #TODO



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrain_holes', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--start_with_holes', action='store_true')
    #parser.add_argument('--variance_reduction', action='store_true')
    parser.add_argument('-k', type=int, default=3) #TODO
    parser.add_argument('--load_data', action='store_true')
    args = parser.parse_args()

    max_length = 30
    batch_size = 100

    train_data = 'data/DeepCoder_data/T3_A2_V512_L10_train_perm.txt'
    lines = (line.rstrip('\n') for i, line in enumerate(open(train_data)) if i != 0) #remove first line



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
        model = SyntaxCheckingRobustFill(input_vocabularies=[list(range(-512, 512)), list(range(-512,512))], target_vocabulary=vocab, max_length=max_length) #TODO

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

            for i, batch in enumerate(batchloader(lines, batchsize=100, N=5, V=512, L=10, compute_sketches=False)):

                print("tic")
                t = time.time()
                score, syntax_score = model.optimiser_step(batch.IOs, batch.pseqs) #TODO make sure inputs are correctly formatted
                print(f"tock, network time: {time.time()-t}, other time: {t-t2}")
                t2 = time.time()
                model.pretrain_scores.append(score)
                model.pretrain_iteration += 1
                if i%1==0: print("pretrain iteration", i, "score:", score, "syntax score:", syntax_score, flush=True)

        #to prevent overwriting model:
        if not args.nosave:
            torch.save(model, './deepcoder_pretrained.p')
    ######## End Pretraining without holes ########



    ####### train with holes ########

    #make this a function (or class, whatever)
    print("training with holes")
    model = model.with_target_vocabulary(vocab) #TODO
    model.cuda()

    if not hasattr(model, 'iteration') or args.start_with_holes:
        model.iteration = 0
    if not hasattr(model, 'hole_scores'):
        model.hole_scores = []
    if args.load_data:
        with open(data_file, 'rb') as file:
            data = pickle.load(file)


    for i in range(model.iteration, 10000): #TODO

        if args.load_data:
            batch = data[i:i+batch_size]
            IO = [datum['IO'] for datum in batch]
            sketchseq = [datum['sketchseq'] for datum in batch]
        else:
            IO, pseq, p, tp = getBatch()
            #assert False
            holey_p = [make_holey_deepcoder(prog, args.k, grammar, request) for prog, request in zip(p, tp)] #TODO
            sketchseq = [flatten_program(prog) for prog in holey_p]

        objective, syntax_score = model.optimiser_step(IO, sketchseq)

        #TODO: also can try RL objective, but unclear why that would be better.

        model.iteration += 1
        model.hole_scores.append(objective)
        if i%1==0: 
            print("iteration", i, "score:", objective, "syntax_score:", syntax_score, flush=True)
        if i%100==0:
            inst = getInstance()
            print("inputs:", inst['IO'])
            samples, scores, _ = model.sampleAndScore(batch_inputs=[inst['IO']], nRepeats=100) #Depending on the model
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







