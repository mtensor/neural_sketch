#demo.py


import argparse
import torch
from torch import nn, optim

from pinn import RobustFill
import pregex as pre
#from vhe import VHE, DataLoader, Factors, Result, RegexPrior
import random

from sketch_project import Hole


parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', action='store_true')
args = parser.parse_args()


print("loading model")

if args.pretrained:
	print("loading pretrained_model")
	model=torch.load("./sketch_model.p")
else:
	print("loading model with holes")
	model=torch.load("./sketch_model_holes.p")


for i in range(999):
	print("-"*20, "\n")
	if i==0:
		examples = ["bar", "car", "dar"]
		print("Using examples:")
		for e in examples: print(e)
		print()
	else:
		print("Please enter examples (one per line):")
		examples = []
		nextInput = True
		while nextInput:
			s = input()
			if s=="":
				nextInput=False
			else:
				examples.append(s)

	print("calculating... ")
	samples, scores = model.sampleAndScore([examples], nRepeats=100)
	print(samples)
	index = scores.index(max(scores))
	#print(samples[index])
	try: sample = pre.create(list(samples[index]))
	except: sample = samples[index]
	#sample = samples[index]
	print("best example by nn score:", sample, ", nn score:", max(scores))


	pregexes = []
	pscores = []
	for samp in samples:
		try: 
			reg = pre.create(list(samp))
			pregexes.append(reg)
			pscores.append(sum(reg.match(ex) for ex in examples )) 
		except:
			continue 

	index = pscores.index(max(pscores))
	preg = pregexes[index] 

	print("best example by pregex score:", preg, ", preg score:", max(pscores))
