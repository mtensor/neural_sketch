#Training robustfill deepcoderModel
from builtins import super
import pickle
import string
import argparse
import random

import torch
from torch import nn, optim

import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./ec'))

from pinn import RobustFill
from pinn import SyntaxCheckingRobustFill #TODO
import random
import math
import time

from collections import OrderedDict
#from util import enumerate_reg, Hole


from grammar import Grammar, NoCandidates
from RobustFillPrimitives import RobustFillProductions, flatten_program
from program import Application, Hole, Primitive, Index, Abstraction, ParseFailure
import math
from type import Context, arrow, tint, tlist, tbool, UnificationFailure
from util.robustfill_util import parseprogram, robustfill_vocab
from data_src.makeRobustFillData import batchloader
from itertools import chain
from models.deepcoderModel import LearnedFeatureExtractor, DeepcoderRecognitionModel
from models.deepcoderModel import RobustFillLearnedFeatureExtractor, load_rb_dc_model_from_path
from models.deepcoderModel import SketchFeatureExtractor, HoleSpecificFeatureExtractor, ImprovedRecognitionModel

from string import printable

#   from deepcoderModel import 
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--nosave', action='store_true')
parser.add_argument('-k', type=int, default=3) #TODO
parser.add_argument('--Vrange', type=int, default=128)
parser.add_argument('--max_list_length', type=int, default=10)
parser.add_argument('--save_model_path', type=str, default='./saved_models/text_dc_model.p')
parser.add_argument('--load_model_path', type=str, default='./saved_models/text_dc_model.p')
parser.add_argument('--new', action='store_true')
parser.add_argument('--n_examples', type=int, default=4)
parser.add_argument('--max_length', type=int, default=25)
parser.add_argument('--max_index', type=int, default=4)
parser.add_argument('--max_iteration', type=int, default=50*400*100) #approximate other model
parser.add_argument('--improved_dc_model', action='store_true')
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--inv_temp', type=float, default=0.1) #idk what the deal with this is ...
parser.add_argument('--use_timeout', action='store_true', default=True)
parser.add_argument('--nHoles', type=int, default=1)
parser.add_argument('--use_dc_grammar', action='store_true')

args = parser.parse_args()

batchsize = 1
max_iteration = args.max_iteration
max_list_length = args.max_list_length    
use_dc_grammar = args.use_dc_grammar
cuda=args.cuda

robustfill_io_vocab = printable[:-4]

basegrammar = Grammar.fromProductions(RobustFillProductions(args.max_length, args.max_index))
vocab = robustfill_vocab(basegrammar)

if __name__ == "__main__":


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
            ###
            specExtractor = RobustFillLearnedFeatureExtractor(robustfill_io_vocab, hidden=128, use_cuda=cuda)
            sketchExtractor = SketchFeatureExtractor(vocab, hidden=128, use_cuda=cuda)
            extractor = HoleSpecificFeatureExtractor(specExtractor, sketchExtractor, hidden=128, use_cuda=cuda)
            dcModel = ImprovedRecognitionModel(extractor, basegrammar, hidden=[128], cuda=cuda, contextual=False)
            ###
        else:
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
        for i, datum in zip(range(max_iteration - dcModel.iteration), 
                        batchloader(max_iteration - dcModel.iteration,
                                                basegrammar,
                                                batchsize=1,
                                                N=args.n_examples,
                                                V=args.max_length,
                                                L=args.max_list_length, 
                                                compute_sketches=args.improved_dc_model,
                                                dc_model=dcModel if use_dc_grammar and (dcModel.epochs > 1) else None, # TODO
                                                improved_dc_model=args.improved_dc_model,
                                                top_k_sketches=args.k,
                                                inv_temp=args.inv_temp,
                                                nHoles=args.nHoles,
                                                use_timeout=args.use_timeout)): #TODO


            t = time.time()
            t3 = t-t2
            if args.improved_dc_model:
                score = dcModel.optimizer_step((datum.IO, datum.sketchseq), datum.p, datum.sketch, datum.tp)
            else:
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
