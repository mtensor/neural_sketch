#evaluate.py


#import statements
import argparse
import torch
from torch import nn, optim

from pinn import RobustFill
#import pregex as pre
#from pregex import ParseException
#from vhe import VHE, DataLoader, Factors, Result, RegexPrior
import random
import math

from sketch_project_rl_regex import Hole

from util import enumerate_reg, fill_hole, date_data, all_data, Hole #TODO change all of these, they are all from regex domain
import time
import pickle

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


def evaluate(model, dataset, nRepeats, mdl, d, pretrained=False):
	t = time.time()

	### run the network ###
	sketches = []
	best_sketch = []
	best_full_prog = []
	best_ll = []

	for task, examples in enumerate(dataset): #ignoring for now #TODO multiple at once
		samples, _ = model.sampleAndScore([examples], nRepeats=nRepeats)

		sketches.append([])
		best_sketch.append( None )
		best_full_prog.append( None )
		best_ll.append( float('-inf') )

		for sample in samples:
			if sample.count(Hole) == 1: #TODO try sketches with more than one hole 
				try:	
					sketches[task].append( pre.create(sample, lookup=d) )
				except ParseException: 
					continue
			elif Hole not in sample:
				#this line prevents recalculating over and over again when there are multiple holes
				try:	
					r = pre.create(sample, lookup=d)
				except ParseException: 
					continue
				ll = sum(r.match(example)/len(example) for example in dataset[task])/len(dataset[task]) #average per character ll
				#ll = sum(r.match(example) for example in dataset[task])
				if ll > best_ll[task]:
					best_sketch[task] = r
					best_full_prog[task] = r
					best_ll[task] = ll



	### enumeration ###
	if not pretrained:
		for i, (subtree, _) in enumerate(enumerate_reg(mdl)): #10 #13
			for task in range(len(dataset)):
				for sketch in sketches[task]:
					full_program = fill_hole(sketch, subtree)
					ll = sum(full_program.match(example)/len(example) for example in dataset[task])/len(dataset[task]) #average per character ll
					#ll = sum(full_program.match(example) for example in dataset[task])
					if ll > best_ll[task]:
						best_sketch[task] = sketch
						best_full_prog[task] = full_program
						best_ll[task] = ll
			if i%100==0:
				print("subtree number:", i)
				if i!=0: print("took {0:.2f} seconds from last iteration".format(time.time() - itime))
				itime = time.time()


	else: #if pretrained model w/out holes
		for task in range(len(dataset)):
			for full_program in sketches[task]:
				#full_program = fill_hole(sketch, subtree)
				ll = sum(full_program.match(example)/len(example) for example in dataset[task])/len(dataset[task]) #average per character ll
				#ll = sum(full_program.match(example) for example in dataset[task])
				if ll > best_ll[task]:
					best_sketch[task] = full_program
					best_full_prog[task] = full_program
					best_ll[task] = ll	


	total_ll = math.log(sum(math.exp(ll) for ll in best_ll))
	task_tuples = zip(best_sketch, best_full_prog, best_ll, dataset)


	print("total_ll (average per character ll) is:", total_ll)
	print("evaluation took {0:.2f} seconds".format(time.time()-t))

	results = {'total_ll': total_ll, 'task_tuples':task_tuples, 'dataset':dataset, 'sketches':sketches, 'best_sketch': best_sketch, 'best_full_prog':best_full_prog, 'best_ll':best_ll}
	return results

def save_results(results, pretrained=False):
	timestr = str(int(time.time()))
	if pretrained:
		filename = "results_pretrain_" + timestr + '.p'
	else:
		filename = "results_" + timestr + '.p'
	with open(filename, 'wb') as savefile:
		pickle.dump(results, savefile)
		print("results file saved")



#fuctions:
	#evaluate
	#save


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--pretrained', action='store_true')
	parser.add_argument('--pretrain_holes', action='store_true')
	args = parser.parse_args()

	nSamples = 100
	mdl = 9 #9 gives 408, 10 gives 1300, 13 gives 24000
	nExamples = 5

	lookup_d = {Hole:Hole()}

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
	results = evaluate(model, dataset, nSamples, mdl, lookup_d, pretrained=args.pretrained)

	#doesn't really need a full function ... 
	save_results(results, pretrained=args.pretrained)



	####cool graphic#####
	for sketch, prog, ll, task in results['task_tuples']:
		print(task, "-->", sketch, "-->", prog, "with ll", ll)

