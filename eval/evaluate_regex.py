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
from util.regex_util import enumerate_reg, fill_hole, date_data, all_data, Hole, basegrammar
import time
import pickle

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
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--pretrain_holes', action='store_true')
parser.add_argument('--max_to_check', type=int, default=10000)
args = parser.parse_args()

RegexResult = namedtuple("RegexResult", ["sketch", "prog", "ll", "n_checked", "time"])

lookup_d = {Hole:Hole()}

nSamples = 100
mdl = 9 #9 gives 408, 10 gives 1300, 13 gives 24000
nExamples = 5
max_to_check = args.max_to_check

# def evaluate(model, dataset, nRepeats, mdl, d, pretrained=False):
# 	t = time.time()
# 	### run the network ###
# 	sketches = []
# 	best_sketch = []
# 	best_full_prog = []
# 	best_ll = []
# 	for task, examples in enumerate(dataset): #ignoring for now #TODO multiple at once
# 		samples, _ = model.sampleAndScore([examples], nRepeats=nRepeats)
# 		sketches.append([])
# 		best_sketch.append( None )
# 		best_full_prog.append( None )
# 		best_ll.append( float('-inf') )
# 		for sample in samples:
# 			if sample.count(Hole) == 1: #TODO try sketches with more than one hole 
# 				try:	
# 					sketches[task].append( pre.create(sample, lookup=d) )
# 				except ParseException: 
# 					continue
# 			elif Hole not in sample:
# 				#this line prevents recalculating over and over again when there are multiple holes
# 				try:	
# 					r = pre.create(sample, lookup=d)
# 				except ParseException: 
# 					continue
# 				ll = sum(r.match(example)/len(example) for example in dataset[task])/len(dataset[task]) #average per character ll
# 				#ll = sum(r.match(example) for example in dataset[task])
# 				if ll > best_ll[task]:
# 					best_sketch[task] = r
# 					best_full_prog[task] = r
# 					best_ll[task] = ll
# 	### enumeration ###
# 	if not pretrained:
# 		for i, (subtree, _) in enumerate(enumerate_reg(mdl)): #10 #13
# 			for task in range(len(dataset)):
# 				for sketch in sketches[task]:
# 					full_program = fill_hole(sketch, subtree)
# 					ll = sum(full_program.match(example)/len(example) for example in dataset[task])/len(dataset[task]) #average per character ll
# 					#ll = sum(full_program.match(example) for example in dataset[task])
# 					if ll > best_ll[task]:
# 						best_sketch[task] = sketch
# 						best_full_prog[task] = full_program
# 						best_ll[task] = ll
# 			if i%100==0:
# 				print("subtree number:", i)
# 				if i!=0: print("took {0:.2f} seconds from last iteration".format(time.time() - itime))
# 				itime = time.time()
# 	else: #if pretrained model w/out holes
# 		for task in range(len(dataset)):
# 			for full_program in sketches[task]:
# 				#full_program = fill_hole(sketch, subtree)
# 				ll = sum(full_program.match(example)/len(example) for example in dataset[task])/len(dataset[task]) #average per character ll
# 				#ll = sum(full_program.match(example) for example in dataset[task])
# 				if ll > best_ll[task]:
# 					best_sketch[task] = full_program
# 					best_full_prog[task] = full_program
# 					best_ll[task] = ll	
# 	total_ll = math.log(sum(math.exp(ll) for ll in best_ll))
# 	task_tuples = zip(best_sketch, best_full_prog, best_ll, dataset)
# 	print("total_ll (average per character ll) is:", total_ll)
# 	print("evaluation took {0:.2f} seconds".format(time.time()-t))
# 	results = {'total_ll': total_ll, 'task_tuples':task_tuples, 'dataset':dataset, 'sketches':sketches, 'best_sketch': best_sketch, 'best_full_prog':best_full_prog, 'best_ll':best_ll}
# 	return results

def alternate(*args):
	# note: python 2 - use izip_longest
	for iterable in zip_longest(*args):
		for item in iterable:
			if item is not None:
				yield item

def test_program_on_IO(e, IO):
	return all(reduce(lambda a, b: a(b), xs, e)==y for xs, y in IO)

def evaluate_datum(i, datum, model, dcModel, nRepeats, mdl, max_to_check):
	t = time.time()
	samples = {("<HOLE>",)}  # make more general
	n_checked, n_hit = 0, 0
	g = basegrammar if not dcModel else dcModel.infer_grammar(datum[:5])  # TODO pp
	if model:
		# can replace with a beam search at some point
		# TODO: use score for allocating resources
		# tokenized = tokenize_for_robustfill([datum])
		samples, _scores = model.sampleAndScore([datum[:5]], nRepeats=nRepeats)  # TODO pp
		# only loop over unique samples:
		samples = {tuple(sample) for sample in samples if sample.count(Hole)==1}  # only 
	sketches = []
	for sample in samples:
		try:
			# sk = parseprogram(sample, datum.tp)  
			sk = pre.create(sample, lookup=lookup_d)
			sketches.append(sk)
		except ParseException:
			n_checked += 1
			yield RegexResult(sample, None, float('-inf'), n_checked, time.time()-t)
			continue
	# only loop over unique sketches:
	sketches = {sk for sk in sketches}
	#alternate which sketch to enumerate from each time
	for _, _, subtree in g.enumeration(Context.EMPTY, [], tpregex, mdl):
		subtree = subtree.evaluate([])
		for sketch in sketches:
			full_program = fill_hole(sketch, subtree)
			ll = sum(full_program.match(example)/len(example) for example in datum[5:10])/len(datum[5:10])  # TODO pp
			prog = full_program if ll > float('-inf') else None
			n_checked += 1
			yield RegexResult(sk, prog, ll, n_checked, time.time()-t)
			if n_checked >= max_to_check: break
	######TODO: want search time and total time to hit task ######
	print(f"task {i}:")
	print(f"evaluation for task {i} took {time.time()-t} seconds")
	print(f"For task {i}, tried {n_checked} sketches, found {n_hit} hits")

def evaluate_dataset(model, dataset, nRepeats, mdl, max_to_check, dcModel=None):
	t = time.time()
	if model is None:
		print("evaluating dcModel baseline")
	return {datum: list(evaluate_datum(i, datum, model, dcModel, nRepeats, mdl, max_to_check)) for i, datum in enumerate(dataset)}

def save_results(results, pretrained=False):
	timestr = str(int(time.time()))
	if pretrained:
		filename = "results_pretrain_" + timestr + '.p'
	else:
		filename = "results_" + timestr + '.p'
	with open(filename, 'wb') as savefile:
		pickle.dump(results, savefile)
		print("results file saved")
	return savefile

def percent_solved_n_checked(results, n_checked):
	return sum(any(result.hit and result.n_checked <= n_checked for result in result_list) for result_list in results.values())/len(results)

def percent_solved_time(results, time):
	return sum(any(result.hit and result.time <= time for result in result_list) for result_list in results.values())/len(results)

if __name__=='__main__':
	#load the model
	if args.pretrained:
		print("loading pretrained_model")
		model = torch.load("./sketch_model_pretrained.p")
	elif args.pretrain_holes:
		print("loading pretrain_holes")
		model = torch.load("./sketch_model_pretrain_holes.p")
	else:
		print("loading model with holes")
		model = torch.load("./experiments/first_fully_sup_1533083502349/sketch_model_holes.p") #TODO

	###load the test dataset###
	# test_dataset = [['mnye@mit.edu','lbh@mit.edu', 'tp@mit.edu']] #['bar','car','dar'], ['m@mit.edu', 'a@mit.edu', 'b@mit.edu'],
	dataset = date_data(20, nExamples=nExamples)
	print("loaded data")

	print("WARNING: using per-character ll evalution, which may not be good...")

	results = evaluate_dataset(model, dataset, nSamples, mdl, max_to_check, dcModel=dcModel)  # TODO
	#results = evaluate(model, dataset, nSamples, mdl, lookup_d, pretrained=args.pretrained)

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

	####cool graphic#####
	for sketch, prog, ll, task in results['task_tuples']:
		print(task, "-->", sketch, "-->", prog, "with ll", ll)

