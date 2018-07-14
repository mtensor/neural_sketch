#demo.py

from sketch_project import *


print("loading model")
#model=torch.load("./sketch_model.p")
model=torch.load("./sketch_model_pretrain_holes.p")


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
