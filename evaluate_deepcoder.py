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

#from sketch_project_rl_regex import Hole

import time
import dill

from main_supervised_deepcoder import parseprogram
from main_supervised_deepcoder import grammar as g

import sys
sys.path.append("/om/user/mnye/ec")

from program import ParseFailure, Context
from grammar import NoCandidates


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


def evaluate(model, dataset, nRepeats, mdl, pretrained=False):
	t = time.time()

	### run the network ###
	sketches = []
	full_prog = []
	hits = []

	for task, datum in enumerate(dataset): #ignoring for now #TODO multiple at once
		t = time.time()
		IO = datum['IO']
		samples, _, _ = model.sampleAndScore([IO], nRepeats=nRepeats) #can replace with a beam search at some point


		#TODO: fix that
		sketches.append([])
		full_prog.append([])
		hits.append([])

		for sample in samples:

				sketches[task].append( sample )
				try:
					sk = parseprogram(sample, datum['tp'])

					# TODO: 
					g = deepcodermodel(IO, g)
					def enumerateFrontiers(self,
										tasks,
										likelihoodModel,
										solver=None,
										enumerationTimeout=None,
										testing=False,
										CPUs=1,
										frontierSize=None,
										maximumFrontier=None,
										evaluationTimeout=None):
						with timing("Evaluated recognition model"):
							grammars = {}
							for task in tasks:
								features = self.featureExtractor.featuresOfTask(task)
								variables, productions = self(features)
									# eprint("variable")
									# eprint(variables.data[0])
									# for k in range(len(self.grammar.productions)):
									# eprint("production",productions.data[k])
								grammars[task] = Grammar(
									variables.data.tolist()[0], [
										(productions.data.tolist()[k], t, p) for k, (_, t, p) in enumerate(
											self.grammar.productions)])


					for l, _, p in g.sketchEnumeration(Context.EMPTY,[], datum['tp'], sk, 12.):
						#check if it works on the IO
						e = p.evaluate([])

						if all(e(x)==y for x, y in IO): #TODO:
							#mark this program as a hit
							hits[task].append(True)
							full_prog[task].append(e)
						else:
							hits[task].append(False)
							full_prog[task].append( None )

							#TODO: include time
				except (ParseFailure, NoCandidates) as e: #TODO
					full_prog[task].append( None )
					hits[task].append(False)
					continue

		print(f"task {task}:")
		print(IO)
		print(f"evaluation for task {task} took {time.time()-t} seconds")
		len(sketches[task])
		print(f"For task {task}, tried {len(sketches[task])} sketches, found {hits[task].count(True)} hits")

		if task==5: break

	results = {'dataset':dataset, 'sketches':sketches, 'hits':hits, 'full_prog':full_prog} #TODO, time evaluation
	return results



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
	parser.add_argument('--pretrain_holes', action='store_true')
	args = parser.parse_args()

	nSamples = 20
	mdl = 9 #9 gives 408, 10 gives 1300, 13 gives 24000
	nExamples = 5

	dataset_file = 'data/pretrain_data_v1.p'

	#load the model
	if args.pretrained:
		print("loading pretrained_model")
		model = torch.load("./deepcoder_model_pretrained.p")
	elif args.pretrain_holes:
		print("loading pretrain_holes")
		model = torch.load("./deepcoder_model_pretrain_holes.p")
	else:
		print("loading model with holes")
		model = torch.load("./deepcoder_holes.p") #TODO

	###load the test dataset###
	# test_dataset = [['mnye@mit.edu','lbh@mit.edu', 'tp@mit.edu']] #['bar','car','dar'], ['m@mit.edu', 'a@mit.edu', 'b@mit.edu'],
	with open(dataset_file, 'rb') as file:
		dataset = dill.load(file)
		print("loaded data")


	results = evaluate(model, dataset, nSamples, mdl, pretrained=args.pretrained)

	#doesn't really need a full function ... 
	save_results(results, pretrained=args.pretrained)


	####cool graphic#####
	for sketch, prog, ll, task in results['task_tuples']:
		print(task, "-->", sketch, "-->", prog, "with ll", ll)

