#Sketch project
from builtins import super
import pickle
import string
import argparse
import random
import torch
from torch import nn, optim
import random
import math
import time
from collections import OrderedDict
from itertools import chain
import math

import sys
import os
#sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./ec'))

from pinn import RobustFill
from pinn import SyntaxCheckingRobustFill #TODO

from grammar import Grammar, NoCandidates
from program import Application, Hole, Primitive, Index, Abstraction, ParseFailure
from type import Context, arrow, tint, tlist, tbool, UnificationFailure
from deepcoderPrimitives import deepcoderProductions, flatten_program
from utilities import timing

from algolispPrimitives import algolispProductions, primitive_lookup, algolisp_input_vocab, algolisp_IO_vocab
from data_src.makeAlgolispData import batchloader
from util.algolisp_util import tokenize_for_robustfill, tree_depth, seq_to_tree, tokenize_IO_for_robustfill

from train.algolisp_train_dc_model import newDcModel
from collections import Counter



# import sys
# sys.path.append("/om/user/mnye/ec")

# from grammar import Grammar, NoCandidates
# from program import Application, Hole, Primitive, Index, Abstraction, ParseFailure
# from type import Context, arrow, tint, tlist, tbool, UnificationFailure
# from deepcoderPrimitives import deepcoderProductions, flatten_program
# from utilities import timing

# from makeDeepcoderData import batchloader
# import math
# from deepcoder_util import parseprogram, grammar, tokenize_for_robustfill
# from itertools import chain

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--nosave', action='store_true')
#parser.add_argument('--start_with_holes', action='store_true')
parser.add_argument('--variance_reduction', action='store_true')
parser.add_argument('--new', action='store_true')
parser.add_argument('--rnn_max_length', type=int, default=250)
parser.add_argument('--batchsize', type=int, default=32)
#parser.add_argument('--n_examples', type=int, default=5)
parser.add_argument('--max_list_length', type=int, default=10)
parser.add_argument('--max_pretrain_epochs', type=int, default=10)
parser.add_argument('--max_pretrain_iterations', type=int, default=10000000)
parser.add_argument('--max_iterations', type=int, default=10000000)
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--train_data', type=str, default='train')
# save and load files
parser.add_argument('--load_pretrained_model_path', type=str, default="./saved_models/algolisp_pretrained.p")
parser.add_argument('--save_pretrained_model_path', type=str, default="./saved_models/algolisp_pretrained.p")
parser.add_argument('--save_model_path', type=str, default="./saved_models/algolisp_holes.p")
parser.add_argument('--save_freq', type=int, default=200)
parser.add_argument('--print_freq', type=int, default=1)
parser.add_argument('-k','--top_k_sketches', type=int, default=100)
parser.add_argument('--inv_temp', type=float, default=1.0)
#parser.add_argument('--use_rl', action='store_true')
#parser.add_argument('--imp_weight_trunc', action='store_true')
#parser.add_argument('--rl_no_syntax', action='store_true')
parser.add_argument('--use_dc_grammar', type=str, default='NA') #'./saved_models/algolisp_dc_model.p') #_5_iter_25500.p
parser.add_argument('--improved_dc_model', action='store_true', default=True)
parser.add_argument('--reward_fn', type=str, default='original', choices=['original','linear','exp', 'flat'])
parser.add_argument('--sample_fn', type=str, default='original', choices=['original','linear','exp', 'flat'])
parser.add_argument('--r_max', type=int, default=8)
parser.add_argument('--timing', action='store_true', default=True)
parser.add_argument('--num_half_lifes', type=float, default=4)
parser.add_argument('--use_timeout', action='store_true')
parser.add_argument('--filter_depth', nargs='+', type=int, default=None)
parser.add_argument('--nHoles', type=int, default=1)

parser.add_argument('--limit_data', type=float, default=False)
parser.add_argument('--train_to_convergence', action='store_true')

parser.add_argument('--convergence_mode', type=str, default='eval')
parser.add_argument('--limit_val_data', type=float, default=0.02)
parser.add_argument('--converge_after', type=int, default=5)

parser.add_argument('--load_trained_model', action='store_true')
parser.add_argument('--load_trained_model_path', type=str, default="./saved_models/algolisp_holes.p")

parser.add_argument('--IO2seq', action='store_true')
args = parser.parse_args()

#assume we want num_half_life half lives to occur by the r_max value ...
#alpha = math.log(2)*args.num_half_lifes/math.exp(args.r_max)
reward_fn = {
            'original': None, 
            'linear': lambda x: max(math.exp(args.r_max) - math.exp(-x), 0)/math.exp(args.r_max),
            'exp': lambda x: math.exp(-alpha*math.exp(-x)),
            'flat': lambda x: 1 if x > -args.r_max else 0
                }[args.reward_fn]
sample_fn = {
            'original': None,
            'linear': lambda x: max(math.exp(args.r_max) - math.exp(-x), 0),
            'exp': lambda x: math.exp(-alpha*math.exp(-x)),
            'flat': lambda x: 1 if x > -args.r_max else 0
                }[args.sample_fn]

batchsize = args.batchsize
train_datas = args.train_data

if args.use_dc_grammar == 'NA':
    use_dc_grammar = False
    dc_grammar_path = None
else:
    use_dc_grammar = True
    dc_model_path = args.use_dc_grammar

#if use_dc_grammar:
improved_dc_model = args.improved_dc_model

vocab = list(primitive_lookup.keys()) + ['(',')', '<HOLE>']
inputvocab = algolisp_input_vocab if not args.IO2seq else algolisp_IO_vocab#TODO

if __name__ == "__main__":
    print("Loading model", flush=True)
    try:
        if args.new: raise FileNotFoundError
        elif args.load_trained_model:
            model=torch.load(args.load_trained_model_path)
            print("loading saved trained model, continuing training")
        else:
            model=torch.load(args.load_pretrained_model_path)
            print('found saved model, loaded pretrained model (without holes)')
            model = model.with_target_vocabulary(vocab)
    except FileNotFoundError:
        print("no saved model, creating new one")
        model = SyntaxCheckingRobustFill(
            input_vocabularies=[inputvocab],
            target_vocabulary=vocab, max_length=args.rnn_max_length, hidden_size=512)
        model.pretrain_iteration = 0
        model.pretrain_scores = []
        model.pretrain_epochs = 0
        model.iteration = 0
        model.hole_scores = []
        model.epochs = 0
        model.pretrain_val_scores = [float('-inf')]
        model.val_scores = [float('-inf')]

    if use_dc_grammar:
        print("loading dc model")
        dc_model=newDcModel(IO2seq=args.IO2seq)
        dc_model.load_state_dict(torch.load(dc_model_path))
        dc_model.cuda()

    model.cuda()
    print("number of parameters is", sum(p.numel() for p in model.parameters() if p.requires_grad))


    # dataset = batchloader(train_datas,
    #                                             batchsize=1,
    #                                             compute_sketches=False,
    #                                             dc_model=dc_model if use_dc_grammar else None,
    #                                             improved_dc_model=improved_dc_model,
    #                                             top_k_sketches=args.top_k_sketches,
    #                                             inv_temp=args.inv_temp,
    #                                             reward_fn=reward_fn,
    #                                             sample_fn=sample_fn,
    #                                             use_timeout=args.use_timeout,
    #                                             filter_depth=args.filter_depth)

    # c = Counter()
    # c.update(tree_depth(seq_to_tree(d.pseq)) for d in dataset)
    # print(c)
    # assert False

    ####### train with holes ########
    pretraining = args.pretrain and model.pretrain_epochs < args.max_pretrain_epochs
    training = model.epochs < args.max_epochs

    t2 = time.time()
    while pretraining or training:
        j = model.pretrain_epochs if pretraining else model.epochs
        if pretraining: print(f"\tpretraining epoch {j}:")
        else: print(f"\ttraining epoch {j}:")
        path = args.save_pretrained_model_path if pretraining else args.save_model_path
        for i, batch in enumerate(batchloader(train_datas,
                                                batchsize=batchsize,
                                                compute_sketches=not pretraining,
                                                dc_model=dc_model if use_dc_grammar else None,
                                                improved_dc_model=improved_dc_model,
                                                top_k_sketches=args.top_k_sketches,
                                                inv_temp=args.inv_temp,
                                                reward_fn=reward_fn,
                                                sample_fn=sample_fn,
                                                use_timeout=args.use_timeout,
                                                filter_depth=args.filter_depth,
                                                nHoles=args.nHoles,
                                                limit_data=args.limit_data)):
            specs = tokenize_for_robustfill(batch.specs) if not args.IO2seq else tokenize_IO_for_robustfill(batch.IOs)
            if i==0: print("batchsize:", len(specs))
            if args.timing: t = time.time()
            objective, syntax_score = model.optimiser_step(specs, batch.pseqs if pretraining else batch.sketchseqs)
            if args.timing:
                print(f"network time: {time.time()-t}, other time: {t-t2}")
                t2 = time.time()
            if pretraining:
                model.pretrain_scores.append(objective)
                model.pretrain_iteration += 1
                if model.pretrain_iteration >= args.max_pretrain_iterations: break
            else:
                model.iteration += 1
                if model.iteration >= args.max_iterations: break
                model.hole_scores.append(objective)
            if i%args.print_freq==0:
                #if args.use_rl: print("reweighted_reward:", reweighted_reward.mean().data.item())
                print("iteration", i, "score:", objective , "syntax_score:", syntax_score, flush=True)
            if i%args.save_freq==0: 
                if not args.nosave:
                    torch.save(model, path+f'_{str(j)}_iter_{str(i)}.p')
                    torch.save(model, path)
        if not args.nosave:
            torch.save(model, path+'_{}.p'.format(str(j)))
            torch.save(model, path)
        if pretraining: model.pretrain_epochs += 1
        else: model.epochs += 1

        #code for determining if training to convergence
        if args.train_to_convergence and j >= args.converge_after: #completed at least 3 epochs ... 
            val_objective = 0
            for batch in batchloader(args.convergence_mode,
                        batchsize=batchsize,
                        compute_sketches=not pretraining,
                        dc_model=dc_model if use_dc_grammar else None,
                        improved_dc_model=improved_dc_model,
                        top_k_sketches=args.top_k_sketches,
                        inv_temp=args.inv_temp,
                        reward_fn=reward_fn,
                        sample_fn=sample_fn,
                        use_timeout=args.use_timeout,
                        filter_depth=args.filter_depth,
                        nHoles=args.nHoles,
                        limit_data=args.limit_val_data,
                        use_fixed_seed=True): #TODO
                specs = tokenize_for_robustfill(batch.specs) if not args.IO2seq else tokenize_IO_for_robustfill(batch.IOs)
                val_objective_iter, _ = model.score(specs, batch.pseqs if pretraining else batch.sketchseqs)
                val_objective += val_objective_iter.mean()
            print("epoch", model.pretrain_epochs if pretraining else model.epochs, "score:", val_objective, flush=True)
            if pretraining:
                model.pretrain_val_scores.append(val_objective)
            else:
                model.val_scores.append(val_objective)

            if val_objective < ( model.pretrain_val_scores[-2] if pretraining else model.val_scores[-2]): #TODO
                if pretraining: pretraining = False
                else: training = False

        #switch from training to pretraining
        if model.pretrain_epochs >= args.max_pretrain_epochs: pretraining = False
        if model.epochs >= args.max_epochs: training = False

        ####### End train with holes ########

# RL formalism w luke
# add temperature parameter - x 
# think about RL objective

# merge pretrain and regular train - x 
# use with timing(nn training) - idk 
# deal with model attributes - x
