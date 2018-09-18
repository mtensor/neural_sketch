#Training deepcoderModel
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
from deepcoderModel import LearnedFeatureExtractor, DeepcoderRecognitionModel

#   from deepcoderModel import 
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--nosave', action='store_true')
parser.add_argument('-k', type=int, default=3) #TODO
parser.add_argument('--Vrange', type=int, default=128)
parser.add_argument('--max_epochs', type=int, default=50)
parser.add_argument('--max_list_length', type=int, default=10)
parser.add_argument('--save_model_path', type=str, default='./dc_model.p')
parser.add_argument('--load_model_path', type=str, default='./dc_model.p')
parser.add_argument('--new', action='store_true')
parser.add_argument('--train_data', nargs='*', 
    default=['data/DeepCoder_data/T2_A2_V512_L10_train.txt', 'data/DeepCoder_data/T3_A2_V512_L10_train_perm.txt'])
args = parser.parse_args()

max_length = 30
batchsize = 1
Vrange = args.Vrange
max_epochs = args.max_epochs
max_list_length = args.max_list_length

def deepcoder_vocab(grammar, n_inputs=3): 
    return [prim.name for prim in grammar.primitives] + ['input_' + str(i) for i in range(n_inputs)] + ['<HOLE>'] #TODO

deepcoder_io_vocab = list(range(-Vrange, Vrange+1)) + ["LIST_START", "LIST_END"]

if __name__ == "__main__":

    train_datas = args.train_data

    def loader():
        return batchloader(train_datas, batchsize=batchsize, N=5, V=Vrange, L=max_list_length, compute_sketches=False, shuffle=True)

    vocab = deepcoder_vocab(grammar)

    print("Loading model", flush=True)
    try:
        if args.new:
            raise FileNotFoundError
        dcModel=torch.load(args.load_model_path)
        print('found saved dcModel, loading ...')
    except FileNotFoundError:
        print("no saved dcModel, creating new one")
        extractor = LearnedFeatureExtractor(deepcoder_io_vocab, hidden=128)
        dcModel = DeepcoderRecognitionModel(extractor, grammar, hidden=[128], cuda=True)

    print("number of parameters is", sum(p.numel() for p in dcModel.parameters() if p.requires_grad))

    ######## TRAINING ########
    #make this a function...
    t2 = time.time()
    print("training", flush=True)
    if not hasattr(dcModel, 'iteration'):
        dcModel.iteration = 0
        dcModel.scores = []
    if not hasattr(dcModel, 'epochs'):
        dcModel.epochs = 0

    for j in range(dcModel.epochs, max_epochs): #TODO
        print(f"\tepoch {j}:")

        for i, datum in enumerate(loader()): #TODO

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
                    torch.save(dcModel, args.save_model_path+f'_{str(j)}_iter_{str(i)}.p')
        #to prevent overwriting model:
        if not args.nosave:
            torch.save(dcModel, args.save_model_path+'_{}.p'.format(str(j)))
            torch.save(dcModel, args.save_model_path)


    ######## End training ########
