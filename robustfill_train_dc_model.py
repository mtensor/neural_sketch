#Training robustfill deepcoderModel
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
from RobustFillPrimitives import RobustFillProductions, flatten_program
from program import Application, Hole, Primitive, Index, Abstraction, ParseFailure
import math
from type import Context, arrow, tint, tlist, tbool, UnificationFailure
from robustfill_util import parseprogram, robustfill_vocab
from makeRobustFillData import batchloader
from itertools import chain
from deepcoderModel import LearnedFeatureExtractor, DeepcoderRecognitionModel, RobustFillLearnedFeatureExtractor

from string import printable

#   from deepcoderModel import 
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--nosave', action='store_true')
parser.add_argument('-k', type=int, default=3) #TODO
parser.add_argument('--Vrange', type=int, default=128)
parser.add_argument('--max_list_length', type=int, default=10)
parser.add_argument('--save_model_path', type=str, default='./rb_dc_model.p')
parser.add_argument('--load_model_path', type=str, default='./rb_dc_model.p')
parser.add_argument('--new', action='store_true')
parser.add_argument('--n_examples', type=int, default=4)
parser.add_argument('--max_length', type=int, default=25)
parser.add_argument('--max_index', type=int, default=4)
parser.add_argument('--max_iteration', type=int, default=50*400*100) #approximate other model
args = parser.parse_args()

batchsize = 1
max_iteration = args.max_iteration
max_list_length = args.max_list_length    

robustfill_io_vocab = printable[:-4]

basegrammar = Grammar.fromProductions(RobustFillProductions(args.max_length, args.max_index))

if __name__ == "__main__":


    print("Loading model", flush=True)
    try:
        if args.new:
            raise FileNotFoundError
        dcModel=torch.load(args.load_model_path)
        print('found saved dcModel, loading ...')
    except FileNotFoundError:
        print("no saved dcModel, creating new one")
        extractor = RobustFillLearnedFeatureExtractor(robustfill_io_vocab, hidden=128)  # probably want to make it much deeper .... 
        dcModel = DeepcoderRecognitionModel(extractor, basegrammar, hidden=[128], cuda=True)  # probably want to make it much deeper .... 

    print("number of parameters is", sum(p.numel() for p in dcModel.parameters() if p.requires_grad))

    ######## TRAINING ########
    #make this a function...
    t2 = time.time()
    print("training", flush=True)
    if not hasattr(dcModel, 'iteration'):
        dcModel.iteration = 0
        dcModel.scores = []

    if dcModel.iteration <= max_iteration:
        for i, datum in zip(range(max_iteration - dcModel.iteration), batchloader(max_iteration - dcModel.iteration,
                                                batchsize=1,
                                                g=basegrammar,
                                                N=args.n_examples,
                                                V=args.max_length,
                                                L=args.max_list_length, 
                                                compute_sketches=False)): #TODO


            t = time.time()
            t3 = t-t2
            score = dcModel.optimizer_step(datum.IO, datum.p, datum.tp) #TODO make sure inputs are correctly formatted
            t2 = time.time()

            dcModel.scores.append(score)
            dcModel.iteration += 1
            if i%500==0 and not i==0:
                print("pretrain iteration", i, "average score:", sum(dcModel.scores[-500:])/500, flush=True)
                print(f"network time: {t2-t}, other time: {t3}")
            if i%50000==0: 
                #do some inference
                #g = dcModel.infer_grammar(IO) #TODO
                if not args.nosave:
                    torch.save(dcModel.state_dict(), args.save_model_path+f'_iter_{str(i)}.p'+'state_dict')
                    torch.save(dcModel.state_dict(), args.save_model_path+'state_dict')
                    #dcModel.load_state_dict(torch.load(args.save_model_path+'state_dict'))
        #to prevent overwriting model:



    ######## End training ########
