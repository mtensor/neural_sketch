#evaluate.py
import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./ec'))

#import statements
import argparse
import torch
from torch import nn, optim
from pinn import RobustFill
import pregex as pre
from pregex import ParseException
import random
import math
from util.regex_util import enumerate_reg, fill_hole, date_data, all_data, PregHole, basegrammar
import time
import pickle

from util.regex_pypy_util import RegexResult, SketchTup, alternate, pypy_enumerate


#make ll per character - done
#TODO:
#make posterior predictive

#train & use dcModel
#which requires converting programs to EC domain

#THREE baselines:
#vanilla RNN
#vanilla dcModel
#neural-sketch w/out dcModel

parser = argparse.ArgumentParser()
parser.add_argument('--n_test', type=int, default=500)
parser.add_argument('--dcModel', action='store_true')
parser.add_argument('--dcModel_path',type=str, default="./saved_models/dc_model.p")
parser.add_argument('--improved_dc_grammar', action='store_true')
parser.add_argument('--dc_baseline', action='store_true')
parser.add_argument('--n_samples', type=int, default=30)
parser.add_argument('--mdl', type=int, default=14)  #9
parser.add_argument('--n_examples', type=int, default=5)
parser.add_argument('--precomputed_data_file', type=str, default='data/prelim_val_data.p')
parser.add_argument('--model_path', type=str, default="./saved_models/deepcoder_holes.p")
parser.add_argument('--max_to_check', type=int, default=5000)
parser.add_argument('--resultsfile', type=str, default='NA')
parser.add_argument('--shuffled', action='store_true')
parser.add_argument('--beam', action='store_true')
args = parser.parse_args()

nSamples = args.n_samples
mdl = args.mdl
nExamples = args.n_examples
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
			samples, _scores = model.beam_decode([datum.spec], beam_size=nRepeats)
		else:
			samples, _scores, _ = model.sampleAndScore([datum.spec], nRepeats=nRepeats)
		# only loop over unique samples:
		samples = {tuple(sample) for sample in samples}  # only 
	if (not improved_dc_grammar) or (not dcModel):
		g = basegrammar if not dcModel else dcModel.infer_grammar(datum.spec)  # TODO pp
		g = untorch(g)
	sketchtups = []
	for sample in samples:
		try:
			sk = tree_to_prog(seq_to_tree(sample))

			if improved_dc_grammar:
				g = untorch(dcModel.infer_grammar((datum.spec, sample))) #TODO: make sure this line is correct ..
			
			sketchtups.append(SketchTup(sk, g))

		except ParseException:
			n_checked += 1
			yield (RegexResult(sample, None, False, n_checked, time.time()-t))
			continue
	# only loop over unique sketches:
	sketchtups = {sk for sk in sketchtups} #fine
	#alternate which sketch to enumerate from each time

	results, n_checked, n_hit = pypy_enumerate(datum.tp, datum.IO, mdl, sketchtups, n_checked, n_hit, t, max_to_check) #should deal with IO
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
	if args.dc_baseline:
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
	dataset = load_dataset()#TODO dataset

	if args.shuffled:
		random.seed(42)
		random.shuffle(dataset)
	#dataset = random.shuffle(dataset)
	del dataset[args.n_test:]

	results = evaluate_dataset(model, dataset, nSamples, mdl, max_to_check, dcModel=dcModel)

	#doesn't really need a full function ... 
	save_results(results, pretrained=args.pretrained)
	file = save_results(results, args)

	# I want a plot of the form: %solved vs n_hits
	x_axis = [10, 20, 50, 100, 200, 400, 600]  # TODO
	y_axis = [percent_solved_n_checked(results, x) for x in x_axis]

	print("percent solved vs number of evaluated programs")
	print("num_checked:", x_axis)
	print("num_solved:", y_axis)

	#doesn't really need a full function ... 
	file = save_results(results, args)



