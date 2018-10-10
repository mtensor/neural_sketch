# evaluate.py
# import statements
import argparse

import torch
from torch import nn, optim
from pinn import RobustFill
import random
import math
import time
import dill
import pickle
from deepcoder_util import grammar as basegrammar
from deepcoder_util import parseprogram, tokenize_for_robustfill
from makeDeepcoderData import batchloader
from manipulate_results import percent_solved_n_checked, percent_solved_time, plot_result

from pypy_util import DeepcoderResult, alternate, pypy_enumerate

# TODO
import sys
sys.path.append("/om/user/mnye/ec")
from program import ParseFailure, Context
from grammar import NoCandidates, Grammar
from utilities import timing, callCompiled

from itertools import islice, zip_longest


""" rough schematic of what needs to be done:
1) Evaluate NN on test inputs
2) Possible beam search to find candidate sketches
3) unflatten "parseprogram" the candidate sketches
4) use hole enumerator from kevin to turn sketch -> program (g.sketchEnumeration)
5) when hit something, record and move on.

so I need to build parseprogram, beam search and that may be it.
"""
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--n_test', type=int, default=500)
parser.add_argument('--dcModel', action='store_true')
parser.add_argument('--dcModel_path',type=str, default="./dc_model.p")
parser.add_argument('--dc_baseline', action='store_true')
parser.add_argument('--n_samples', type=int, default=30)
parser.add_argument('--mdl', type=int, default=14)  #9
parser.add_argument('--n_examples', type=int, default=5)
parser.add_argument('--Vrange', type=int, default=128)
parser.add_argument('--precomputed_data_file', type=str, default='data/prelim_val_data.p')
parser.add_argument('--model_path', type=str, default="./deepcoder_holes.p")
parser.add_argument('--max_to_check', type=int, default=5000)
parser.add_argument('--resultsfile', type=str, default='NA')
parser.add_argument('--shuffled', action='store_true')
parser.add_argument('--beam', action='store_true')
args = parser.parse_args()

nSamples = args.n_samples
mdl = args.mdl
nExamples = args.n_examples
Vrange = args.Vrange
max_to_check = args.max_to_check

def untorch(g):
	return Grammar(g.logVariable.data.tolist()[0], 
                               [ (l.data.tolist()[0], t, p)
                                 for l, t, p in g.productions])

def evaluate_datum(i, datum, model, dcModel, nRepeats, mdl, max_to_check):
	t = time.time()
	samples = {("<HOLE>",)}  # make more general
	n_checked, n_hit = 0, 0
	g = basegrammar if not dcModel else dcModel.infer_grammar(datum.IO)
	if model:
		# can replace with a beam search at some point
		# TODO: use score for allocating resources
		tokenized = tokenize_for_robustfill([datum.IO])
		if args.beam:
			samples, _scores = model.beam_decode(tokenized, beam_size=nRepeats)
		else:
			samples, _scores, _ = model.sampleAndScore(tokenized, nRepeats=nRepeats)
		# only loop over unique samples:
		samples = {tuple(sample) for sample in samples}
	sketches = []
	for sample in samples:
		try:
			sk = parseprogram(sample, datum.tp)
			sketches.append(sk)
		except (ParseFailure, NoCandidates) as e:
			n_checked += 1
			yield DeepcoderResult(sample, None, False, n_checked, time.time()-t)
			continue
	# only loop over unique sketches:
	sketches = {sk for sk in sketches}
	print(len(sketches))
	print(sketches)
	#alternate which sketch to enumerate from each time

	results, n_checked, n_hit = pypy_enumerate(untorch(g), datum.tp, datum.IO, mdl, sketches, n_checked, n_hit, t, max_to_check)
	yield from (result for result in results)

	######TODO: want search time and total time to hit task ######
	print(f"task {i}:")
	print(f"evaluation for task {i} took {time.time()-t} seconds")
	print(f"For task {i}, tried {n_checked} sketches, found {n_hit} hits", flush=True)

def evaluate_dataset(model, dataset, nRepeats, mdl, max_to_check, dcModel=None):
	t = time.time()
	if model is None:
		print("evaluating dcModel baseline")
	return {datum: list(evaluate_datum(i, datum, model, dcModel, nRepeats, mdl, max_to_check)) for i, datum in enumerate(dataset)}

#TODO: refactor for strings
def save_results(results, args):
	timestr = str(int(time.time()))
	r = '_test' + str(args.n_test) + '_'
	if args.resultsfile != 'NA':
		filename = 'results/' + args.resultsfile + '.p'
	elif args.dc_baseline:
		filename = "results/prelim_results_dc_baseline_" + r + timestr + '.p'
	elif args.pretrained:
		filename = "results/prelim_results_rnn_baseline_" + r + timestr + '.p'
	else:
		dc = 'wdcModel_' if args.dcModel else ''
		filename = "results/prelim_results_" + dc + r + timestr + '.p'
	with open(filename, 'wb') as savefile:
		dill.dump(results, savefile)
		print("results file saved at", filename)
	return savefile


if __name__=='__main__':
	#load the model
	if args.pretrained:
		print("loading pretrained_model")
		model = torch.load("./deepcoder_pretrained.p")
	elif args.dc_baseline:
		print("computing dc baseline, no model")
		assert args.dcModel
		model = None
	else:
		print("loading model with holes")
		model = torch.load(args.model_path) #TODO
	if args.dcModel:
		print("loading dc_model")
		dcModel = torch.load(args.dcModel_path)
	else: dcModel = None

	###load the test dataset###
	# test_data = ['data/DeepCoder_test_data/T3_A2_V512_L10_P500.txt']
	# test_data = ['data/DeepCoder_test_data/T5_A2_V512_L10_P100_test.txt'] #modified from original
	# test_data = ['data/DeepCoder_data/T3_A2_V512_L10_validation_perm.txt']
	# dataset = batchloader(test_data, batchsize=1, N=5, V=Vrange, L=10, compute_sketches=False)
	# dataset = list(dataset)
	# with open('data/prelim_val_data.p', 'wb') as savefile:
	# 	pickle.dump(dataset, savefile)
	# 	print("test file saved")

	print("data file:", args.precomputed_data_file)
	with open(args.precomputed_data_file, 'rb') as datafile:
		dataset = pickle.load(datafile)
	# optional:

	if args.shuffled:
		random.seed(42)
		random.shuffle(dataset)
	#dataset = random.shuffle(dataset)
	del dataset[args.n_test:]

	results = evaluate_dataset(model, dataset, nSamples, mdl, max_to_check, dcModel=dcModel)

	# count hits
	hits = sum(any(result.hit for result in result_list) for result_list in results.values())
	print(f"hits: {hits} out of {args.n_test}, or {100*hits/args.n_test}% accuracy")

	# I want a plot of the form: %solved vs n_hits
	x_axis = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800, 900, 1000, 2000, 4000]  # TODO
	y_axis = [percent_solved_n_checked(results, x) for x in x_axis]

	print("percent solved vs number of evaluated programs")
	print("num_checked:", x_axis)
	print("num_solved:", y_axis)

	#doesn't really need a full function ... 
	file = save_results(results, args)

	plot_result(results=results, plot_time=True, model_path=args.model_path) #doesn't account for changing result thingy