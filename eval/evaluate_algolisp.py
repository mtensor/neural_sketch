#evaluate.py
import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./ec'))

#import statements
import argparse
import torch
from torch import nn, optim
import random
import math
import time
import pickle
import dill

from util.algolisp_pypy_util import AlgolispResult, SketchTup, alternate, pypy_enumerate, algolisp_enumerate #TODO
from util.algolisp_util import tokenize_for_robustfill, seq_to_tree, tree_to_prog  # TODO
from data_src.makeAlgolispData import batchloader, basegrammar
from train.algolisp_train_dc_model import newDcModel

from plot.manipulate_results import percent_solved_n_checked, percent_solved_time, plot_result


from grammar import Grammar
from itertools import islice
#train & use dcModel
#which requires converting programs to EC domain
parser = argparse.ArgumentParser()
parser.add_argument('--n_test', type=int, default=15)
parser.add_argument('--dcModel', action='store_true', default=True)
parser.add_argument('--dcModel_path',type=str, default="./saved_models/algolisp_dc_model.p")
parser.add_argument('--improved_dc_grammar', action='store_true', default=True)
parser.add_argument('--dc_baseline', action='store_true')
parser.add_argument('--n_samples', type=int, default=30)
parser.add_argument('--mdl', type=int, default=14)  #9
parser.add_argument('--n_examples', type=int, default=5)
parser.add_argument('--model_path', type=str, default="./saved_models/algolisp_pretrained.p")
parser.add_argument('--max_to_check', type=int, default=5000)
parser.add_argument('--resultsfile', type=str, default='first_rnn_algolisp_results')
parser.add_argument('--shuffled', action='store_true')
parser.add_argument('--beam', action='store_true', default=True)
parser.add_argument('--dataset', type=str, default='eval')
parser.add_argument('--pypy',action='store_true')
args = parser.parse_args()

nSamples = args.n_samples
mdl = args.mdl
nExamples = args.n_examples #TODO
max_to_check = args.max_to_check
improved_dc_grammar = args.improved_dc_grammar
if improved_dc_grammar: assert args.dcModel


def untorch(g):
	if type(g.logVariable) == float:
		return g
	else:
		return Grammar(g.logVariable.data.tolist()[0], 
                               [ (l.data.tolist()[0], t, p)
                                 for l, t, p in g.productions])

def evaluate_datum(i, datum, model, dcModel, nRepeats, mdl, max_to_check):
	t = time.time()
	samples = {('<HOLE>',)}  # make more general #  TODO, i don't think 
	n_checked, n_hit = 0, 0
	if model:
		if args.beam:
			samples, _scores = model.beam_decode(tokenize_for_robustfill([datum.spec]), beam_size=nRepeats)
		else:
			samples, _scores, _ = model.sampleAndScore(tokenize_for_robustfill([datum.spec]), nRepeats=nRepeats)
		# only loop over unique samples:
		samples = {tuple(sample) for sample in samples}  # only 
	if (not improved_dc_grammar) or (not dcModel):
		g = basegrammar if not dcModel else dcModel.infer_grammar(datum.spec)  # TODO pp
		g = untorch(g)
	sketchtups = []
	for sample in samples:
		try:
			sk = tree_to_prog(seq_to_tree(sample))
		except Exception as e: # TODO: needs to be fixed
			print("EXCEPTION IN PARSE:,", e)
			n_checked += 1
			yield (AlgolispResult(sample, None, False, n_checked, time.time()-t))
			continue

		if improved_dc_grammar:
			g = untorch(dcModel.infer_grammar((datum.spec, sample))) #TODO: make sure this line is correct ..
		
		sketchtups.append(SketchTup(sk, g))


	# only loop over unique sketches:
	sketchtups = {sk for sk in sketchtups} #fine
	print("sketchtups:", sketchtups)
	#alternate which sketch to enumerate from each time

	if args.pypy:
		results, n_checked, n_hit = pypy_enumerate(datum.tp, datum.IO, datum.schema_args, mdl, sketchtups, n_checked, n_hit, t, max_to_check)
	else:
		results, n_checked, n_hit = algolisp_enumerate(datum.tp, datum.IO, datum.schema_args, mdl, sketchtups, n_checked, n_hit, t, max_to_check) #might need more than IO
	yield from (result for result in results)

	######TODO: want search time and total time to hit task ######
	print(f"task {i}:")
	print(f"evaluation for task {i} took {time.time()-t} seconds")
	print(f"For task {i}, tried {n_checked} sketches, found {n_hit} hits")

def evaluate_dataset(model, dataset, nRepeats, mdl, max_to_check, dcModel=None):
	t = time.time()
	if model is None:
		print("evaluating dcModel baseline")
	return {datum: list(evaluate_datum(i, datum, model, dcModel, nRepeats, mdl, max_to_check) ) for i, datum in enumerate(dataset)}

def save_results(results, args):
	timestr = str(int(time.time()))
	r = '_test' + str(args.n_test) + '_'
	if args.resultsfile != 'NA':
		filename = 'algolisp_results/' + args.resultsfile + '.p'
	elif args.dc_baseline:
		filename = "algolisp_results/algolisp_prelim_results_dc_baseline_" + r + timestr + '.p'
	elif args.pretrained:
		filename = "algolisp_results/algolisp_prelim_results_rnn_baseline_" + r + timestr + '.p'
	else:
		dc = 'wdcModel_' if args.dcModel else ''
		filename = "algolisp_results/algolisp_prelim_results_" + dc + r + timestr + '.p'
	with open(filename, 'wb') as savefile:
		dill.dump(results, savefile)
		print("results file saved at", filename)
	return savefile

if __name__=='__main__':
	#load the model
	if args.dc_baseline:
		print("computing dc baseline, no model")
		assert args.dcModel
		model = None
	else:
		print("loading model with holes")
		model = torch.load(args.model_path) #TODO
		model.cuda()
	if args.dcModel:
		print("loading dcModel")
		dcModel=newDcModel()
		dcModel.load_state_dict(torch.load(args.dcModel_path))
		dcModel.cuda()
	else: dcModel = None

	###load the test dataset###
	dataset = batchloader(args.dataset, batchsize=1,
                                                compute_sketches=False,
                                                dc_model=None,
                                                improved_dc_model=False)  #TODO
	dataset = islice(dataset, args.n_test) #TODO

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

	file = save_results(results, args)

	plot_result(results=results, plot_time=True, model_path=args.model_path) #doesn't account for changing result thingy

