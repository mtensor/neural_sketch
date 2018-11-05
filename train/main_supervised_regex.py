#Sketch project regex supervised
from builtins import super
import pickle
import string
import argparse
import random
import torch
from torch import nn, optim
import random
import math

import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./ec'))

from pinn import RobustFill
import pregex as pre

from collections import OrderedDict
from util.regex_util import enumerate_reg, PregHole, tokenize_for_robustfill, basegrammar

from data_src.makeRegexData import batchloader


regex_vocab = list(string.printable[:-4]) + \
    [pre.OPEN, pre.CLOSE, pre.String, pre.Concat, pre.Alt, pre.KleeneStar, pre.Plus, pre.Maybe, PregHole] + \
    regex_prior.character_classes


parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--nosave', action='store_true')
#parser.add_argument('--start_with_holes', action='store_true')
#parser.add_argument('--variance_reduction', action='store_true')
parser.add_argument('-k', type=int, default=50) #TODO
parser.add_argument('--new', action='store_true')
parser.add_argument('--rnn_max_length', type=int, default=30)
parser.add_argument('--batchsize', type=int, default=200)
#parser.add_argument('--Vrange', type=int, default=128)
parser.add_argument('--n_examples', type=int, default=5)
#parser.add_argument('--max_list_length', type=int, default=10)
#parser.add_argument('--max_n_inputs', type=int, default=3)
#parser.add_argument('--max_pretrain_epochs', type=int, default=10)
parser.add_argument('--max_pretrain_iterations', type=int, default=100000)
parser.add_argument('--max_iterations', type=int, default=100000)
#parser.add_argument('--max_epochs', type=int, default=10)
#parser.add_argument('--train_data', nargs='*', 
#    default=['data/DeepCoder_data/T2_A2_V512_L10_train.txt', 'data/DeepCoder_data/T3_A2_V512_L10_train_perm.txt'])
# save and load files
parser.add_argument('--load_pretrained_model_path', type=str, default="./saved_models/regex_pretrained.p")
parser.add_argument('--save_pretrained_model_path', type=str, default="./saved_models/regex_pretrained.p")
parser.add_argument('--save_model_path', type=str, default="./saved_models/regex_holes.p")
parser.add_argument('--save_freq', type=int, default=200)
parser.add_argument('--print_freq', type=int, default=1)
parser.add_argument('--top_k_sketches', type=int, default=100)
parser.add_argument('--inv_temp', type=float, default=1.0)
#parser.add_argument('--use_rl', action='store_true')
#parser.add_argument('--imp_weight_trunc', action='store_true')
#parser.add_argument('--rl_no_syntax', action='store_true')
parser.add_argument('--use_dc_grammar', type=str, default='NA')
#parser.add_argument('--rl_lr', type=float, default=0.001)
parser.add_argument('--reward_fn', type=str, default='original', choices=['original','linear','exp', 'flat'])
parser.add_argument('--sample_fn', type=str, default='original', choices=['original','linear','exp', 'flat'])
parser.add_argument('--r_max', type=int, default=8)
parser.add_argument('--timing', action='store_true')
#parser.add_argument('--num_half_lifes', type=float, default=4)
parser.add_argument('--use_timeout', action='store_true')
###
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--pretrain_holes', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--nosave', action='store_true')
parser.add_argument('--start_with_holes', action='store_true')
parser.add_argument('--enum_dict', type=int, default=1000)
###
args = parser.parse_args()

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
#Vrange = args.Vrange
#train_datas = args.train_data

if args.use_dc_grammar == 'NA':
    use_dc_grammar = False
    dc_grammar_path = None
else:
    use_dc_grammar = True
    dc_model_path = args.use_dc_grammar

if __name__ == "__main__":

    print("Loading model", flush=True)
    try:
        if args.new: raise FileNotFoundError
        else:
            model=torch.load(args.load_pretrained_model_path)
            print('found saved model, loaded pretrained model (without holes)')
    except FileNotFoundError:
        print("no saved model, creating new one")
        model = SyntaxCheckingRobustFill(input_vocabularies=[string.printable[:-4]], target_vocabulary=regex_vocab, max_length=args.rnn_max_length)

        model.pretrain_iteration = 0
        model.pretrain_scores = []
        model.iteration = 0
        model.hole_scores = []

    if use_dc_grammar:
        print("loading dc model")
        dc_model=torch.load(dc_model_path)

    model.cuda()
    print("number of parameters is", sum(p.numel() for p in model.parameters() if p.requires_grad))

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
                                                batchsize=batchsize,
                                                g=basegrammar,
                                                N=args.n_examples, 
                                                compute_sketches=not pretraining,
                                                dc_model=dc_model if use_dc_grammar else None,
                                                top_k_sketches=args.top_k_sketches,
                                                inv_temp=args.inv_temp,
                                                reward_fn=reward_fn,
                                                sample_fn=sample_fn,
                                                use_timeout=args.use_timeout)):
            IOs = tokenize_for_robustfill(batch.IOs)
            if args.timing: t = time.time()
            objective, syntax_score = model.optimiser_step(IOs, batch.pseqs if pretraining else batch.sketchseqs)
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
                print("iteration", i, "score:", objective if not args.use_rl else score.mean().data.item() , "syntax_score:", syntax_score if not args.use_rl else syntax_score.data.item(), flush=True)
            if i%args.save_freq==0: 
                if not args.nosave:
                    torch.save(model, path+f'_iter_{str(i)}.p')
                    torch.save(model, path)
        if model.pretrain_iteration >= args.max_pretrain_iteration: pretraining = False
        if not pretraining and model.iteration < args.max_iteration: training = True
        if training and model.iteration >= args.max_iteration: training = False



# ###############
#         if args.start_with_holes:
#             assert False
#             model=torch.load('')
#             print('found saved model, loading pretrained model with holes')
#         else:
#             model=torch.load(load_pretrained_model_path)
#             print('found saved model, loading pretrained model (without holes)')

#     except FileNotFoundError:
#         print("no saved model, creating new one")
#         model = RobustFill(input_vocabularies=[string.printable[:-4]], target_vocabulary=regex_vocab, max_length=max_length)

#     model.cuda()
#     print("number of parameters is", sum(p.numel() for p in model.parameters() if p.requires_grad))


#     #######create the enum_dict ######
#     print("creating the enum_dict")

#     d = {r: s for r, s in enumerate_reg(13)} #TODO #13 gives 24888
#     #sort dict
#     enum_dict = OrderedDict(sorted(d.items(), key=lambda s: -s[1]))
#     print("size of enum dict is", len(enum_dict))


#     ######## Pretraining without holes ########

#     def getInstance(k_shot=4):
#         """
#         Returns a single problem instance, as input/target strings
#         """
#         #k_shot = 4 #random.choice(range(3,6)) #this means from 3 to 5 examples
 
#         while True:
#             r = regex_prior.sampleregex()
#             c = r.flatten()
#             x = r.sample()
#             Dc = [r.sample() for i in range(k_shot)]
#             c_input = [c]
#             if all(len(x)<max_length for x in Dc + [c, x]): break
#         return {'Dc':Dc, 'c':c, 'c_input':c_input, 'x':x, 'r':r}

#     def getBatch():
#         """
#         Create a batch of problem instances, as tensors
#         """
#         k_shot = random.choice(range(3,6)) #this means from 3 to 5 examples

#         instances = [getInstance(k_shot=k_shot) for i in range(batch_size)]
#         Dc = [inst['Dc'] for inst in instances]
#         c = [inst['c'] for inst in instances]
#         c_input = [inst['c_input'] for inst in instances]
#         x = [inst['x'] for inst in instances]
#         r = [inst['r'] for inst in instances]
#         return Dc, c, c_input, x, r 

#     if args.pretrain:

#         print("pretraining", flush=True)
#         if not hasattr(model, 'pretrain_iteration'):
#             model.pretrain_iteration = 0
#             model.pretrain_scores = []

#         if args.debug:
#             Dc, c, _, _, _ = getBatch()

#         for i in range(model.pretrain_iteration, 20000):
#             if not args.debug:
#                 Dc, c, _, _, _ = getBatch()
#             #print("Dc:", Dc)
#             #print("c:", c)
#             score = model.optimiser_step(Dc, c)

#             model.pretrain_scores.append(score)
#             model.pretrain_iteration += 1
#             if i%10==0: print("pretrain iteration", i, "score:", score, flush=True)

#             #to prevent overwriting model:
#             if not args.nosave:
#                 if i%500==0: torch.save(model, './sketch_model.p')
#     ######## End Pretraining without holes ########


#     max_length = 30
#     batch_size = 100

#     ####### train with holes ########
#     print("training with holes")
#     model = model.with_target_vocabulary(regex_vocab)
#     model.cuda()

#     if args.variance_reduction:
#         if not hasattr(model, 'variance_red'):
#             model.variance_red = nn.Parameter(torch.Tensor([1])).cuda()

#     if not args.pretrain_holes:
#         optimizer = optim.Adam(model.parameters(), lr=1e-2)

#     if not hasattr(model, 'iteration') or args.start_with_holes:
#         model.iteration = 0
#     if not hasattr(model, 'hole_scores'):
#         model.hole_scores = []
#     for i in range(model.iteration, 10000):

#         if not args.pretrain_holes:
#             optimizer.zero_grad()
#         Dc, c, _, _, r = getBatch()

#         #TODO: deal with this line:
#         holey_r, holescore = zip(*[ make_holey_supervised(reg, enum_dict, k=1)[0] for reg in r]) # i think this is the line i have to change 

#         sketch = [reg.flatten() for reg in holey_r]

#         if not args.pretrain_holes:
#             objective = model.score(Dc, sketch, autograd=True)
#             objective = objective.mean()
#             #print(objective)
#             (-objective).backward()
#             optimizer.step()

#         else: #if args.pretrain_holes:
#             objective = model.optimiser_step(Dc, sketch)

#         model.iteration += 1
#         model.hole_scores.append(objective)
#         if i%1==0: 
#             print("iteration", i, "score:", objective, flush=True)
#         if i%100==0:
#             inst = getInstance()
#             samples, scores = model.sampleAndScore([inst['Dc']], nRepeats=100)
#             index = scores.index(max(scores))
#             #print(samples[index])
#             try: sample = pre.create(list(samples[index]))
#             except: sample = samples[index]
#             sample = samples[index]
#             print("actual program:", pre.create(inst['c']))
#             print("generated examples:")
#             print(*inst['Dc'])
#             print("inferred:", sample)

#         if i%100==0: # and not i==0: 
#             if not args.nosave:
#                 torch.save(model, './saved_models/sketch_model_holes_ep_{}.p'.format(str(i)))
#                 torch.save(model, './saved_models/sketch_model_holes.p')

#     ####### End train with holes ########






