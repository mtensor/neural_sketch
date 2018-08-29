#evaluate.py


#import statements
import argparse
from collections import namedtuple
import torch
from torch import nn, optim

from pinn import RobustFill
#import pregex as pre
#from pregex import ParseException
#from vhe import VHE, DataLoader, Factors, Result, RegexPrior
import random
import math

#from sketch_project_rl_regex import Hole

import time
import dill

from main_supervised_deepcoder import tokenize_for_robustfill

from deepcoder_util import grammar as basegrammar
from deepcoder_util import parseprogram, 

import sys
sys.path.append("/om/user/mnye/ec")

from program import ParseFailure, Context
from grammar import NoCandidates

from itertools import islice

#make ll per character - done
#TODO: make posterior predictive - IDK what the correct eval criterion should be here.
#MODELING: make model with RNN syntax filter - done


""" rough schematic of what needs to be done:
1) Evaluate NN on test inputs
2) Possible beam search to find candidate sketches
3) unflatten "parseprogram" the candidate sketches
4) use hole enumerator from kevin to turn sketch -> program (g.sketchEnumeration)
5) when hit something, record and move on.


so I need to build parseprogram, beam search and that may be it.
"""

DeepcoderResult = namedtuple("DeepcoderResult", ["sketch", "prog", "hit"])

def test_program_on_IO(e, IO):
	assert False
	return all(e(x)==y for x, y in IO) # TODO

def evaluate_datum(i, datum, model, dcModel, nRepeats, mdl):
	t = time.time()
	samples = [["<HOLE>"]]  # make more general
	n_checked, n_hit = 0, 0
	if model:
		# can replace with a beam search at some point
		# TODO: use score for allocating resources
		tokenized = tokenize_for_robustfill([datum.IO])
		samples, _score, _ = model.sampleAndScore(tokenized, nRepeats=nRepeats)
	for sample in samples:
		try:
			sk = parseprogram(sample, datum.tp)
		except (ParseFailure, NoCandidates) as e:
			n_checked += 1
			yield DeepcoderResult(sample, None, False)
			continue
		g = basegrammar if not dcModel else dcModel.infer_grammar(datum.IO)
		for l, _, p in g.sketchEnumeration(Context.EMPTY,[], datum.tp, sk, mdl):
			e = p.evaluate_dataset([])
			hit = test_program_on_IO(e, datum.IO)
			prog = e if hit else None
			n_checked += 1
			n_hit += 1 if hit else 0
			yield DeepcoderResult(sample, prog, hit)
	######TODO: want search time and total time to hit task ######
	print(f"task {i}:")
	print(IO)
	print(f"evaluation for task {i} took {time.time()-t} seconds")
	print(f"For task {i}, tried {n_checked} sketches, found {n_hit} hits")

def evaluate_dataset(model, dataset, nRepeats, mdl, dcModel=None):
	t = time.time()
	if model is None:
		print("evaluating dcModel baseline")
	return {datum: list(evaluate_datum(i, datum, model, dcModel, nRepeats, mdl)) for i, datum in enumerate(dataset)}


#TODO:
def save_results(results, pretrained=False):
	timestr = str(int(time.time()))
	if pretrained:
		filename = "results_pretrain_" + timestr + '.p'
	else:
		filename = "results_" + timestr + '.p'
	with open(filename, 'wb') as savefile:
		dill.dump(results, savefile)
		print("results file saved")

#fuctions:
	#evaluate
	#save

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--pretrained', action='store_true')
	parser.add_argument('--n_test', type=int, default=20)
	args = parser.parse_args()

	nSamples = 20
	mdl = 9 #9 gives 408, 10 gives 1300, 13 gives 24000
	nExamples = 5

	Vrange = 128

	dataset_file = 'data/pretrain_data_v1.p'

	#load the model
	if args.pretrained:
		print("loading pretrained_model")
		model = torch.load("./deepcoder_model_pretrained.p")
	elif args.dc_baseline:
		print("computing dc baseline, no model")
		assert args.dcModel
	else:
		print("loading model with holes")
		model = torch.load("./deepcoder_holes.p") #TODO


	if args.dcModel:
		print("loading dc_model")
		dcModel = torch.load("./dc_model.p")


	###load the test dataset###
    test_data = 'data/DeepCoder_test_data/T3_A2_V512_L10_P500.txt'
    lines = (line.rstrip('\n') for i, line in enumerate(open(train_data)) if i != 0) #remove first line
    dataset = batchloader(lines, batchsize=1, N=5, V=Vrange, L=10, compute_sketches=False):
    
    #optional:
    dataset = list(dataset)
    dataset = random.shuffle(dataset)
    del dataset[args.n_test:]

	results = evaluate_dataset(model, dataset, nSamples, mdl, dcModel=dcModel)

	#doesn't really need a full function ... 
	save_results(results, pretrained=args.pretrained)


	####cool graphic#####

