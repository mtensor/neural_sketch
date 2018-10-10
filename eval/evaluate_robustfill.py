# evaluate.py
import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./ec'))
# import statements
import argparse
from collections import namedtuple
import torch
from torch import nn, optim
from pinn import RobustFill
import random
import math
import time
import dill
import pickle
from util.robustfill_util import basegrammar
from util.robustfill_util import parseprogram, tokenize_for_robustfill
from data.makeRobustFillData import batchloader, Datum
from plot.manipulate_results import percent_solved_n_checked, percent_solved_time, plot_result
from models.deepcoderModel import load_rb_dc_model_from_path, LearnedFeatureExtractor, DeepcoderRecognitionModel, RobustFillLearnedFeatureExtractor

from util.rb_pypy_util import RobustFillResult, rb_pypy_enumerate

from util.pypy_util import alternate


from program import ParseFailure, Context
from grammar import NoCandidates, Grammar

from itertools import islice, zip_longest
from functools import reduce

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
parser.add_argument('--pretrained_model_path', type=str, default="./robustfill_pretrained.p")
parser.add_argument('--n_test', type=int, default=500)
parser.add_argument('--dcModel', action='store_true')
parser.add_argument('--dc_model_path', type=str, default="./robustfill_dc_model.p")
parser.add_argument('--dc_baseline', action='store_true')
parser.add_argument('--n_samples', type=int, default=30)
parser.add_argument('--mdl', type=int, default=17)  #9
parser.add_argument('--n_rec_examples', type=int, default=4)
parser.add_argument('--max_length', type=int, default=25)
parser.add_argument('--max_index', type=int, default=4)
parser.add_argument('--precomputed_data_file', type=str, default='rb_test_tasks.p')
parser.add_argument('--model_path', type=str, default="./robustfill_holes.p")
parser.add_argument('--max_to_check', type=int, default=5000)
parser.add_argument('--resultsfile', type=str, default='NA')
parser.add_argument('--test_generalization', action='store_true')
parser.add_argument('--beam', action='store_true')
args = parser.parse_args()

nSamples = args.n_samples
mdl = args.mdl
n_rec_examples = args.n_rec_examples

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
		tokenized = tokenize_for_robustfill([datum.IO[:n_rec_examples]])
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
			yield RobustFillResult(sample, None, False, n_checked, time.time()-t, False)
			continue
	# only loop over unique sketches:
	sketches = {sk for sk in sketches}
	print(len(sketches))
	print(sketches)
	#alternate which sketch to enumerate from each time
	results, n_checked, n_hit = rb_pypy_enumerate(untorch(g), datum.tp, datum.IO, mdl, sketches, n_checked, n_hit, t, max_to_check, args.test_generalization, n_rec_examples)
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
		filename = 'rb_results/' + args.resultsfile + '.p'
	elif args.dc_baseline:
		filename = "rb_results/prelim_results_dc_baseline_" + r + timestr + '.p'
	elif args.pretrained:
		filename = "rb_results/prelim_results_rnn_baseline_" + r + timestr + '.p'
	else:
		dc = 'wdcModel_' if args.dcModel else ''
		filename = "rb_results/prelim_results_" + dc + r + timestr + '.p'
	with open(filename, 'wb') as savefile:
		dill.dump(results, savefile)
		print("results file saved at", filename)
	return savefile


if __name__=='__main__':
	#load the model
	if args.pretrained:
		print("loading pretrained model")
		model = torch.load(args.pretrained_model_path)
	elif args.dc_baseline:
		print("computing dc baseline, no model")
		assert args.dcModel
		model = None
	else:
		print("loading model with holes")
		model = torch.load(args.model_path) #TODO
	if args.dcModel:
		print("loading dc_model")
		dcModel = load_rb_dc_model_from_path(args.dc_model_path, args.max_length, args.max_index)


	print("data file:", args.precomputed_data_file)
	with open(args.precomputed_data_file, 'rb') as datafile:
		dataset = pickle.load(datafile)
	# optional:
	#dataset = random.shuffle(dataset)
	del dataset[args.n_test:]

	results = evaluate_dataset(model, dataset, nSamples, mdl, max_to_check, dcModel=dcModel if args.dcModel else None)

	# count hits
	hits = sum(any(result.hit for result in result_list) for result_list in results.values())
	print(f"hits: {hits} out of {len(dataset)}, or {100*hits/len(dataset)}% accuracy")

	# I want a plot of the form: %solved vs n_hits
	x_axis = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800, 900, 1000, 2000, 4000]  # TODO
	y_axis = [percent_solved_n_checked(results, x) for x in x_axis]

	print("percent solved vs number of evaluated programs")
	print("num_checked:", x_axis)
	print("num_solved:", y_axis)

	#doesn't really need a full function ... 
	file = save_results(results, args)

	plot_result(results=results, plot_time=True, model_path=args.model_path, robustfill=True) #doesn't account for changing result thingy