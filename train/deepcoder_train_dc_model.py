#Training deepcoderModel
from builtins import super
import pickle
import string
import argparse
import random

import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./ec'))
import torch
from torch import nn, optim

from pinn import RobustFill
from pinn import SyntaxCheckingRobustFill #TODO
import random
import math
import time

from collections import OrderedDict
from itertools import chain

from grammar import Grammar, NoCandidates
from deepcoderPrimitives import deepcoderProductions, flatten_program
from program import Application, Hole, Primitive, Index, Abstraction, ParseFailure
from type import Context, arrow, tint, tlist, tbool, UnificationFailure
from util.deepcoder_util import parseprogram, basegrammar, deepcoder_vocab
from data_src.makeDeepcoderData import batchloader
from models.deepcoderModel import LearnedFeatureExtractor, DeepcoderRecognitionModel, SketchFeatureExtractor, HoleSpecificFeatureExtractor, ImprovedRecognitionModel


#   from deepcoderModel import 
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--nosave', action='store_true')
parser.add_argument('-k', type=int, default=50) #TODO
parser.add_argument('--Vrange', type=int, default=128)
parser.add_argument('--max_epochs', type=int, default=50)
parser.add_argument('--max_list_length', type=int, default=10)
parser.add_argument('--save_model_path', type=str, default='./saved_models/list_dc_model.p')
parser.add_argument('--load_model_path', type=str, default='./saved_models/list_dc_model.p')
parser.add_argument('--new', action='store_true')
parser.add_argument('--max_n_inputs', type=int, default=2)
parser.add_argument('--improved_dc_model', action='store_true')
parser.add_argument('--train_data', nargs='*', 
    default=['data/DeepCoder_data/T2_A2_V512_L10_train.txt', 'data/DeepCoder_data/T3_A2_V512_L10_train_perm.txt'])
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--inv_temp', type=float, default=0.1) #idk what the deal with this is ...
parser.add_argument('--use_timeout', action='store_true', default=True)
parser.add_argument('--nHoles', type=int, default=1)
parser.add_argument('--use_dc_grammar', action='store_true')
parser.add_argument('--max_iterations', type=int, default=100000000)
args = parser.parse_args()

max_length = 30
batchsize = 1
Vrange = args.Vrange
max_epochs = args.max_epochs
max_list_length = args.max_list_length
cuda=args.cuda
use_dc_grammar=args.use_dc_grammar

deepcoder_io_vocab = list(range(-Vrange, Vrange+1)) + ["LIST_START", "LIST_END"]

if __name__ == "__main__":

    train_datas = args.train_data

    vocab = deepcoder_vocab(basegrammar, n_inputs=args.max_n_inputs)

    print("Loading model", flush=True)
    try:
        if args.new:
            raise FileNotFoundError
        dcModel=torch.load(args.load_model_path)
        print('found saved dcModel, loading ...')
    except FileNotFoundError:
        print("no saved dcModel, creating new one")
        if args.improved_dc_model:
            print("creating new improved dc model")
            specExtractor = LearnedFeatureExtractor(deepcoder_io_vocab, hidden=128, use_cuda=cuda)
            sketchExtractor = SketchFeatureExtractor(vocab, hidden=128, use_cuda=cuda)
            extractor = HoleSpecificFeatureExtractor(specExtractor, sketchExtractor, hidden=128, use_cuda=cuda)
            dcModel = ImprovedRecognitionModel(extractor, basegrammar, hidden=[128], cuda=cuda, contextual=False)
        else:
            extractor = LearnedFeatureExtractor(deepcoder_io_vocab, hidden=128)
            dcModel = DeepcoderRecognitionModel(extractor, basegrammar, hidden=[128], cuda=True)

    print("number of parameters is",
        sum(p.numel() for p in dcModel.parameters() if p.requires_grad))

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

        for i, datum in enumerate(
            batchloader(train_datas,
                            batchsize=batchsize,
                            N=5,
                            V=Vrange,
                            L=max_list_length,
                            compute_sketches=args.improved_dc_model,
                            dc_model=dcModel if use_dc_grammar and (dcModel.epochs > 1) else None, # TODO
                            improved_dc_model=args.improved_dc_model,
                            top_k_sketches=args.k,
                            inv_temp=args.inv_temp,
                            reward_fn=None,
                            sample_fn=None,
                            nHoles=args.nHoles,
                            use_timeout=args.use_timeout,
                            shuffle=True)): #TODO

            t = time.time()
            t3 = t-t2
            if args.improved_dc_model:
                score = dcModel.optimizer_step((datum.IO, datum.sketchseq), datum.p, datum.sketch, datum.tp)
            else:
                score = dcModel.optimizer_step(datum.IO, datum.p, datum.tp) #TODO make sure inputs are correctly formatted
            
            t2 = time.time()
            #print(datum.sketch)
            dcModel.scores.append(score)
            dcModel.iteration += 1
            if dcModel.iteration > args.max_iterations:
                print('done training')
                break
            if i%500==0 and not i==0:
                print("pretrain iteration", i, "average score:", sum(dcModel.scores[-500:])/500, flush=True)
                print(f"network time: {t2-t}, other time: {t3}")
            if i%50000==0: 
                #do some inference
                #g = dcModel.infer_grammar(IO) #TODO
                if not args.nosave:
                    torch.save(dcModel, args.save_model_path+f'_{str(j)}_iter_{str(i)}.p')
                    torch.save(dcModel, args.save_model_path)

        #to prevent overwriting model:
        if not args.nosave:
            torch.save(dcModel, args.save_model_path+'_{}.p'.format(str(j)))
            torch.save(dcModel, args.save_model_path)


    ######## End training ########
