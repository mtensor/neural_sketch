#Sketch project
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
from program import Application, Hole, Primitive, Index, Abstraction, ParseFailure
from type import Context, arrow, tint, tlist, tbool, UnificationFailure
from RobustFillPrimitives import RobustFillProductions, flatten_program
from utilities import timing


from models.deepcoderModel import LearnedFeatureExtractor, DeepcoderRecognitionModel, RobustFillLearnedFeatureExtractor, load_rb_dc_model_from_path
from data_src.makeRobustFillData import batchloader
import math
from util.robustfill_util import parseprogram, tokenize_for_robustfill, robustfill_vocab
from itertools import chain
from string import printable

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--nosave', action='store_true')
#parser.add_argument('--start_with_holes', action='store_true')
parser.add_argument('--variance_reduction', action='store_true')
parser.add_argument('-k', type=int, default=50) #TODO
parser.add_argument('--new', action='store_true')
parser.add_argument('--rnn_max_length', type=int, default=20)  # TODO
parser.add_argument('--batchsize', type=int, default=100)
parser.add_argument('--max_length', type=int, default=25)
parser.add_argument('--n_examples', type=int, default=4)
parser.add_argument('--max_list_length', type=int, default=10)
parser.add_argument('--max_index',type=int, default=4)
# parser.add_argument('--max_pretrain_epochs', type=int, default=4)
# parser.add_argument('--max_epochs', type=int, default=5)
parser.add_argument('--max_pretrain_iteration', type=int, default=400*5*2)
parser.add_argument('--max_iteration', type=int, default=400*5*2)
parser.add_argument('--train_data',type=str,default='NA')
# save and load files
parser.add_argument('--load_pretrained_model_path', type=str, default="./saved_models/robustfill_pretrained.p")
parser.add_argument('--save_pretrained_model_path', type=str, default="./saved_models/robustfill_pretrained.p")
parser.add_argument('--save_model_path', type=str, default="./saved_models/robustfill_holes.p")
parser.add_argument('--save_freq', type=int, default=200)
parser.add_argument('--print_freq', type=int, default=1)
parser.add_argument('--top_k_sketches', type=int, default=100)
parser.add_argument('--inv_temp', type=float, default=1.0)
parser.add_argument('--use_rl', action='store_true')
#parser.add_argument('--imp_weight_trunc', action='store_true')
#parser.add_argument('--rl_no_syntax', action='store_true')
parser.add_argument('--use_dc_grammar', type=str, default='NA')
parser.add_argument('--rl_lr', type=float, default=0.001)
parser.add_argument('--reward_fn', type=str, default='original', choices=['original','linear','exp', 'flat'])
parser.add_argument('--sample_fn', type=str, default='original', choices=['original','linear','exp', 'flat'])
parser.add_argument('--r_max', type=int, default=8)
parser.add_argument('--timing', action='store_true')
parser.add_argument('--num_half_lifes', type=float, default=4)
parser.add_argument('--use_timeout', action='store_true')
args = parser.parse_args()

#assume we want num_half_life half lives to occur by the r_max value ...

alpha = math.log(2)*args.num_half_lifes/math.exp(args.r_max)

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
max_length = args.max_length
if args.train_data != 'NA':
    train_datas = args.train_data
    assert False

if args.use_dc_grammar == 'NA':
    use_dc_grammar = False
    dc_grammar_path = None
else:
    use_dc_grammar = True
    dc_model_path = args.use_dc_grammar

basegrammar = Grammar.fromProductions(RobustFillProductions(args.max_length, args.max_index))



vocab = robustfill_vocab(basegrammar)

if __name__ == "__main__":
    print("Loading model", flush=True)
    try:
        if args.new: raise FileNotFoundError
        else:
            model=torch.load(args.load_pretrained_model_path)
            print('found saved model, loaded pretrained model (without holes)')
    except FileNotFoundError:
        print("no saved model, creating new one")
        model = SyntaxCheckingRobustFill(
            input_vocabularies=[printable[:-4], printable[:-4]], 
            target_vocabulary=vocab, max_length=args.rnn_max_length, hidden_size=512)  # TODO
        model.pretrain_iteration = 0
        model.pretrain_scores = []
        model.iteration = 0
        model.hole_scores = []

    if use_dc_grammar:
        print("loading dc model")
        dc_model = load_rb_dc_model_from_path(dc_model_path, args.max_length, args.max_index)

    model.cuda()
    print("number of parameters is", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.use_rl: model._get_optimiser(lr=args.rl_lr)
    if args.variance_reduction:
        #if not hasattr(model, 'variance_red'):
         #   print("creating variance_red param")
            #variance_red = nn.Parameter(torch.Tensor([0], requires_grad=True, device="cuda"))
            #variance_red = torch.zeros(1, requires_grad=True, device="cuda") 
        variance_red = torch.Tensor([.95]).cuda().requires_grad_()
        model.opt.add_param_group({"params": variance_red})
            #model._clear_optimiser()

    ####### train with holes ########
    pretraining = args.pretrain and model.pretrain_iteration < args.max_pretrain_iteration
    training = model.iteration < args.max_iteration and not pretraining

    t2 = time.time()
    while pretraining or training:
        path = args.save_pretrained_model_path if pretraining else args.save_model_path
        iter_remaining = args.max_pretrain_iteration - model.pretrain_iteration if pretraining else args.max_iteration - model.iteration
        print("pretraining:", pretraining)
        print("iter to train:", iter_remaining)
        for i, batch in zip(range(iter_remaining), batchloader(iter_remaining,
                                                g=basegrammar,
                                                batchsize=batchsize,
                                                N=args.n_examples,
                                                V=max_length,
                                                L=args.max_list_length, 
                                                compute_sketches=not pretraining,
                                                dc_model=dc_model if use_dc_grammar else None,
                                                top_k_sketches=args.top_k_sketches,
                                                inv_temp=args.inv_temp,
                                                reward_fn=reward_fn,
                                                sample_fn=sample_fn,
                                                use_timeout=args.use_timeout)):
            IOs = tokenize_for_robustfill(batch.IOs)
            if args.timing: t = time.time()
            if not pretraining and args.use_rl:
                #if not hasattr(model, 'opt'):
                #    model._get_optimiser(lr=args.rl_lr) #todo
                #    model.opt.add_param_group({"params": variance_red})
                model.opt.zero_grad()
                if args.imp_weight_trunc:
                    print("not finished implementing")
                    assert False
                else: 
                    score, syntax_score = model.score(IOs, batch.sketchseqs, autograd=True)
                    print("rewards:", batch.rewards.mean())
                    print("sketchprobs:", batch.sketchprobs.mean())
                    if args.variance_reduction:
                        if not args.rl_no_syntax:
                            objective = torch.exp(score.data)/batch.sketchprobs.cuda() * (batch.rewards.cuda() -  variance_red.data) * (score + syntax_score) - torch.pow((batch.rewards.cuda() - variance_red),2)
                        else:
                            objective = torch.exp(score.data)/batch.sketchprobs.cuda() * (batch.rewards.cuda() -  variance_red.data) * score - torch.pow((batch.rewards.cuda() - variance_red),2)
                        reweighted_reward = torch.exp(score.data)/batch.sketchprobs.cuda() * batch.rewards.cuda()
                    else:
                        if not args.rl_no_syntax:
                            reweighted_reward = torch.exp(score.data)/batch.sketchprobs.cuda() * batch.rewards.cuda()
                            objective = reweighted_reward * (score + syntax_score)
                        else:
                            reweighted_reward = torch.exp(score.data)/batch.sketchprobs.cuda() * batch.rewards.cuda()
                            objective = reweighted_reward * score
                objective = objective.mean()
                (-objective).backward()
                model.opt.step()
                #for the purpose of printing:
                syntax_score = syntax_score.mean()
                objective 
                if args.variance_reduction:
                    print("variance_red_baseline:", variance_red.data.item())
            else:
                objective, syntax_score = model.optimiser_step(IOs, batch.pseqs if pretraining else batch.sketchseqs)
            if args.timing:
                print(f"network time: {time.time()-t}, other time: {t-t2}", flush=True)
                t2 = time.time()
            if pretraining:
                model.pretrain_scores.append(objective)
                model.pretrain_iteration += 1
            else:
                model.iteration += 1
                model.hole_scores.append(objective)
            if i%args.print_freq==0:
                if args.use_rl: print("reweighted_reward:", reweighted_reward.mean().data.item())
                print("iteration", i, "score:", objective if not args.use_rl else score.mean().data.item() , "syntax_score:", syntax_score if not args.use_rl else syntax_score.data.item(), flush=True)
            if i%args.save_freq==0: 
                if not args.nosave:
                    torch.save(model, path+f'_iter_{str(i)}.p')
                    torch.save(model, path)
        if model.pretrain_iteration >= args.max_pretrain_iteration: pretraining = False
        if not pretraining and model.iteration < args.max_iteration: training = True
        if training and model.iteration >= args.max_iteration: training = False

        ####### End train with holes ########

# RL formalism w luke
# add temperature parameter - x 
# think about RL objective

# merge pretrain and regular train - x 
# use with timing(nn training) - idk 
# deal with model attributes - x
