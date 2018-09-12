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
from program import Application, Hole, Primitive, Index, Abstraction, ParseFailure
from type import Context, arrow, tint, tlist, tbool, UnificationFailure
from deepcoderPrimitives import deepcoderProductions, flatten_program
from utilities import timing

from makeDeepcoderData import batchloader
import math
from deepcoder_util import parseprogram, grammar, tokenize_for_robustfill
from itertools import chain

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--nosave', action='store_true')
#parser.add_argument('--start_with_holes', action='store_true')
parser.add_argument('--variance_reduction', action='store_true')
parser.add_argument('-k', type=int, default=50) #TODO
parser.add_argument('--new', action='store_true')
parser.add_argument('--rnn_max_length', type=int, default=30)
parser.add_argument('--batchsize', type=int, default=200)
parser.add_argument('--Vrange', type=int, default=128)
parser.add_argument('--n_examples', type=int, default=5)
parser.add_argument('--max_list_length', type=int, default=10)
parser.add_argument('--max_n_inputs', type=int, default=3)
parser.add_argument('--max_pretrain_epochs', type=int, default=4)
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--train_data', nargs='*', 
    default=['data/DeepCoder_data/T2_A2_V512_L10_train.txt', 'data/DeepCoder_data/T3_A2_V512_L10_train_perm.txt'])
# save and load files
parser.add_argument('--load_pretrained_model_path', type=str, default="./deepcoder_pretrained.p")
parser.add_argument('--save_pretrained_model_path', type=str, default="./deepcoder_pretrained.p")
parser.add_argument('--save_model_path', type=str, default="./deepcoder_holes.p")
parser.add_argument('--save_freq', type=int, default=200)
parser.add_argument('--print_freq', type=int, default=1)
parser.add_argument('--top_k_sketches', type=int, default=100)
parser.add_argument('--inv_temp', type=float, default=1.0)
parser.add_argument('--use_rl', action='store_true')
parser.add_argument('--imp_weight_trunc', action='store_true')
parser.add_argument('--rl_no_syntax', action='store_true')
parser.add_argument('--use_dc_grammar', type=str, default='NA')
parser.add_argument('--rl_lr', type=float, default=0.001)
parser.add_argument('--reward_fn', type=str, default='original', choices=['original','linear','exp'])
parser.add_argument('--sample_fn', type=str, default='original', choices=['original','linear','exp'])
parser.add_argument('--r_max', type=int, default=8)
parser.add_argument('--timing', action='store_true')
parser.add_argument('--num_half_lifes', type=float, default=4)
args = parser.parse_args()

#assume we want num_half_life half lives to occur by the r_max value ...

alpha = math.log(2)*args.num_half_lifes/math.exp(args.r_max)

reward_fn = {
            'original': None, 
            'linear': lambda x: max(math.exp(args.r_max) - math.exp(-x), 0)/math.exp(args.r_max),
            'exp': lambda x: math.exp(-alpha*math.exp(-x))
                }[args.reward_fn]
sample_fn = {
            'original': None,
            'linear': lambda x: max(math.exp(args.r_max) - math.exp(-x), 0),
            'exp': lambda x: math.exp(-alpha*math.exp(-x))
                }[args.sample_fn]


batchsize = args.batchsize 
Vrange = args.Vrange
train_datas = args.train_data

if args.use_dc_grammar == 'NA':
    use_dc_grammar = False
    dc_grammar_path = None
else:
    use_dc_grammar = True
    dc_model_path = args.use_dc_grammar


def deepcoder_vocab(grammar, n_inputs=args.max_n_inputs): 
    return [prim.name for prim in grammar.primitives] + ['input_' + str(i) for i in range(n_inputs)] + ['<HOLE>']  # TODO
vocab = deepcoder_vocab(grammar)

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
            input_vocabularies=[list(range(-Vrange, Vrange+1)) + ["LIST_START", "LIST_END"],
            list(range(-Vrange, Vrange+1)) + ["LIST_START", "LIST_END"]], 
            target_vocabulary=vocab, max_length=args.rnn_max_length, hidden_size=512)  # TODO
        model.pretrain_iteration = 0
        model.pretrain_scores = []
        model.pretrain_epochs = 0
        model.iteration = 0
        model.hole_scores = []
        model.epochs = 0

    if use_dc_grammar:
        print("loading dc model")
        dc_model=torch.load(dc_model_path)

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
                                                N=args.n_examples,
                                                V=Vrange,
                                                L=args.max_list_length, 
                                                compute_sketches=not pretraining,
                                                dc_model=dc_model if use_dc_grammar else None,
                                                top_k_sketches=args.top_k_sketches,
                                                inv_temp=args.inv_temp,
                                                reward_fn=reward_fn,
                                                sample_fn=sample_fn)):
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
                #     batch.rewards, batch.sketchseqs
                #     torch.clamp(q/p, max=c)
                #     pscore, syntax_score = model.score(IOs, batch.sketchseqs, autograd=True)
                #     torch.clamp(qscore/batch.sketchprobs, max=c)
                #     term_1 = torch.clamp(torch.exp(pscore.data)/batch.sketchprobs.cuda(), max=c) * (batch.reward.cuda() -  variance_red.data) * pscore
                #     target, qscore, _ = model.sampleAndScore(IOs, autograd=True)
                #     term_2_rewards = find_rewards(target, )
                #     p_over_q = something#a pain in the but 
                #     term_2 = torch.clamp() * (batch.reward.cuda() -  variance_red.data) * qscore 
                #     objective = term_1 + term_2 + syntax_score - torch.pow((batch.reward.cuda() - variance_red),2)
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
                    #if not args.rl_no_syntax:
                    #    objective = objective + syntax_score
                objective = objective.mean()
                (-objective).backward()
                model.opt.step()
                #for the purpose of printing:
                syntax_score = syntax_score.mean()
                objective 
                #print("objective", objective)
                if args.variance_reduction:
                    print("variance_red_baseline:", variance_red.data.item())
            else:
                objective, syntax_score = model.optimiser_step(IOs, batch.pseqs if pretraining else batch.sketchseqs)
            if args.timing:
                print(f"network time: {time.time()-t}, other time: {t-t2}")
                t2 = time.time()
            if pretraining:
                model.pretrain_scores.append(objective)
                model.pretrain_iteration += 1
            else:
                model.iteration += 1
                model.hole_scores.append(objective)
            if i%args.print_freq==0:
                if args.use_rl: print("reweighted_reward:", reweighted_reward.mean().data.item())
                print("iteration", i, "score:", objective.data.item() if not args.use_rl else score.mean().data.item() , "syntax_score:", syntax_score.data.item(), flush=True)
            if i%args.save_freq==0: 
                if not args.nosave:
                    torch.save(model, path+f'_{str(j)}_iter_{str(i)}.p')
                    torch.save(model, path)
        if not args.nosave:
            torch.save(model, path+'_{}.p'.format(str(j)))
            torch.save(model, path)
        if pretraining: model.pretrain_epochs += 1
        else: model.epochs += 1
        if model.pretrain_epochs >= args.max_pretrain_epochs: pretraining = False
        if model.epochs >= args.max_epochs: training = False

        ####### End train with holes ########

# RL formalism w luke
# add temperature parameter - x 
# think about RL objective

# merge pretrain and regular train - x 
# use with timing(nn training) - idk 
# deal with model attributes - x
