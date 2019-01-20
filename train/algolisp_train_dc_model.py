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
from algolispPrimitives import algolispProductions, primitive_lookup, algolisp_input_vocab, algolisp_IO_vocab
from program import Application, Hole, Primitive, Index, Abstraction, ParseFailure
from type import Context, arrow, tint, tlist, tbool, UnificationFailure
#from util.deepcoder_util import parseprogram, grammar
from data_src.makeAlgolispData import batchloader, basegrammar

from models.deepcoderModel import SketchFeatureExtractor, HoleSpecificFeatureExtractor, ImprovedRecognitionModel, AlgolispIOFeatureExtractor

#   from deepcoderModel import 


def newDcModel(cuda=True, IO2seq=False):
    if IO2seq:
        input_vocab =  algolisp_IO_vocab()# TODO
        algolisp_vocab =  list(primitive_lookup.keys()) + ['(',')', '<HOLE>']
        specExtractor = AlgolispIOFeatureExtractor(input_vocab, hidden=128, use_cuda=cuda) # Is this okay? max length
        sketchExtractor = SketchFeatureExtractor(algolisp_vocab, hidden=128, use_cuda=cuda)
        extractor = HoleSpecificFeatureExtractor(specExtractor, sketchExtractor, hidden=128, use_cuda=cuda)
        dcModel = ImprovedRecognitionModel(extractor, basegrammar, hidden=[128], cuda=cuda, contextual=False)
    else:
        input_vocab =  algolisp_input_vocab# TODO
        algolisp_vocab =  list(primitive_lookup.keys()) + ['(',')', '<HOLE>']
        specExtractor = SketchFeatureExtractor(input_vocab, hidden=128, use_cuda=cuda) # Is this okay? max length
        sketchExtractor = SketchFeatureExtractor(algolisp_vocab, hidden=128, use_cuda=cuda)
        extractor = HoleSpecificFeatureExtractor(specExtractor, sketchExtractor, hidden=128, use_cuda=cuda)
        dcModel = ImprovedRecognitionModel(extractor, basegrammar, hidden=[128], cuda=cuda, contextual=False)


    return(dcModel)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('-k', type=int, default=40) #TODO
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--save_model_path', type=str, default='./saved_models/algolisp_dc_model.p')
    parser.add_argument('--load_model_path', type=str, default='./saved_models/algolisp_dc_model.p')
    parser.add_argument('--new', action='store_true')
    parser.add_argument('--train_data', type=str, default='train')
    parser.add_argument('--use_dc_grammar', action='store_true')
    parser.add_argument('--improved_dc_model', action='store_true', default=True)
    parser.add_argument('--inv_temp', type=float, default=0.01) #idk what the deal with this is ...
    parser.add_argument('--use_timeout', action='store_true', default=True)
    parser.add_argument('--filter_depth', nargs='+', type=int, default=None)
    parser.add_argument('--nHoles', type=int, default=1)
    parser.add_argument('--limit_data', type=float, default=False)
    parser.add_argument('--IO2seq', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_dataset_len', type=int, default=False)

    parser.add_argument('--exclude_odd', action='store_true')
    parser.add_argument('--exclude_even', action='store_true')
    parser.add_argument('--exclude_geq', action='store_true')
    parser.add_argument('--exclude_gt', action='store_true')
    args = parser.parse_args()

    assert not (args.exclude_even and args.exclude_odd)

    #xor all the options classes:
    if any([args.exclude_even, args.exclude_odd, args.exclude_geq]):
        assert (args.exclude_even or args.exclude_odd) != args.exclude_geq

    if args.exclude_odd:
        exclude = [ ["lambda1", ["==", ["%", "arg1", "2"], "1"]] ]
    elif args.exclude_even: 
        exclude = [ ["lambda1", ["==", ["%", "arg1", "2"], "0"]] ] 
    elif args.exclude_geq:
        exclude = [">="]
    elif args.exclude_gt:
        exclude = [">"]
    else: 
        exclude = None

    batchsize = 1
    max_epochs = args.max_epochs
    use_dc_grammar = args.use_dc_grammar
    improved_dc_model = args.improved_dc_model
    top_k_sketches = args.k
    inv_temp = args.inv_temp
    use_timeout = args.use_timeout

    if not improved_dc_model: assert False, "unimplemented"


    train_datas = args.train_data

    print("Loading model", flush=True)
    try:
        if args.new:
            raise FileNotFoundError
        dcModel=newDcModel(IO2seq=args.IO2seq)
        dcModel.load_state_dict(torch.load(args.load_model_path))
        print('found saved dcModel, loading ...')
    except FileNotFoundError:
        print("no saved dcModel, creating new one")


        #extractor = LearnedFeatureExtractor(deepcoder_io_vocab, hidden=128)
        #dcModel = DeepcoderRecognitionModel(extractor, grammar, hidden=[128], cuda=True)

        dcModel = newDcModel(IO2seq=args.IO2seq)

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

        for i, datum in enumerate(batchloader(train_datas,
                                                batchsize=1,
                                                compute_sketches=True,
                                                dc_model=dcModel if use_dc_grammar and (dcModel.epochs > 1) else None, # This means dcModel updated every epoch, but not first two
                                                improved_dc_model=improved_dc_model,
                                                top_k_sketches=args.k,
                                                inv_temp=args.inv_temp,
                                                reward_fn=None,
                                                sample_fn=None,
                                                nHoles=args.nHoles,
                                                use_timeout=args.use_timeout,
                                                filter_depth=args.filter_depth,
                                                limit_data=args.limit_data,
                                                seed=args.seed,
                                                use_dataset_len=args.use_dataset_len,
                                                exclude=exclude)): #TODO

            spec = datum.spec if not args.IO2seq else datum.IO
            t = time.time()
            t3 = t-t2
            #score = dcModel.optimizer_step(datum.IO, datum.p, datum.tp) #TODO make sure inputs are correctly formatted
            score = dcModel.optimizer_step((spec, datum.sketchseq), datum.p, datum.sketch, datum.tp)
            t2 = time.time()

            dcModel.scores.append(score)
            dcModel.iteration += 1
            if i%500==0 and not i==0:
                print("iteration", i, "average score:", sum(dcModel.scores[-500:])/500, flush=True)
                print(f"network time: {t2-t}, other time: {t3}")
            #if i%5000==0:
                if not args.nosave:
                    torch.save(dcModel.state_dict(), args.save_model_path+f'_{str(j)}_iter_{str(i)}.p')
                    torch.save(dcModel.state_dict(), args.save_model_path)
        #to prevent overwriting model:
        dcModel.epochs += 1
        if not args.nosave:
            torch.save(dcModel.state_dict(), args.save_model_path+'_{}.p'.format(str(j)))
            torch.save(dcModel.state_dict(), args.save_model_path)


    ######## End training ########
