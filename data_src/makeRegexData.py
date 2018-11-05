#generate deepcoder data
import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./ec'))

import pickle
# TODO
#from util.deepcoder_util import make_holey_deepcoder # this might be enough
from util.regex_util import basegrammar # TODO
from util.regex_util import sample_program, generate_IO_examples, flatten_program, make_holey_regex
from util.robustfill_util import timing # TODO

import time
from collections import namedtuple
#Function = namedtuple('Function', ['src', 'sig', 'fun', 'bounds'])

from grammar import Grammar, NoCandidates
from utilities import flatten

# TODO
from regexPrimitives import concatPrimitives

from program import Application, Hole, Primitive, Index, Abstraction, ParseFailure
import math
import random
from type import Context, arrow, tint, tlist, UnificationFailure, tcharacter, tpregex
from itertools import zip_longest, chain, repeat, islice
from functools import reduce
import torch


class Datum():
	def __init__(self, tp, p, pseq, IO, sketch, sketchseq, reward, sketchprob):
		self.tp = tp
		self.p = p
		self.pseq = pseq
		self.IO = IO
		self.sketch = sketch
		self.sketchseq = sketchseq
		self.reward = reward
		self.sketchprob = sketchprob

	def __hash__(self): 
		return reduce(lambda a, b: hash(a + hash(b)), flatten(self.IO, abort=lambda x: type(x) is str), 0) + hash(self.p) + hash(self.sketch)

Batch = namedtuple('Batch', ['tps', 'ps', 'pseqs', 'IOs', 'sketchs', 'sketchseqs', 'rewards', 'sketchprobs'])


#TODO#####
def sample_datum(g=basegrammar, N=5, compute_sketches=False, top_k_sketches=100, inv_temp=1.0, reward_fn=None, sample_fn=None, dc_model=None, use_timeout=False, continuation=False):

	# find tp
	if continuation:
		tp = arrow(tpregex, tpregex)
	else:
		tp = tpregex

	#sample a program:
	#with timing("sample program"):
	program = sample_program(g, tp) 

	# find IO
	#with timing("sample IO:"):
	IO = generate_IO_examples(program, num_examples=N, continuation=continuation)  
	if IO is None: return None
	IO = tuple(IO)

	# TODO

	# find pseq
	pseq = tuple(flatten_program(program, continuation=continuation)) #TODO

	if compute_sketches:
		# find sketch

		# TODO - improved dc_grammar [ ]
		# TODO - contextual_grammar [ ]
		# TODO - put grammar inference inside make_holey
		grammar = g if not dc_model else dc_model.infer_grammar(IO)
		
		#with timing("make_holey"):
		sketch, reward, sketchprob = make_holey_regex(program, top_k_sketches, grammar, tp, inv_temp=inv_temp, reward_fn=reward_fn, sample_fn=sample_fn, use_timeout=use_timeout) #TODO

		# find sketchseq
		sketchseq = tuple(flatten_program(sketch, continuation=continuation))
	else:
		sketch, sketchseq, reward, sketchprob = None, None, None, None

	return Datum(tp, program, pseq, IO, sketch, sketchseq, reward, sketchprob)


def grouper(iterable, n, fillvalue=None):
	# "Collect data into fixed-length chunks or blocks"
	# grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
	args = [iter(iterable)] * n
	return zip_longest(*args, fillvalue=fillvalue)

#TODO########
def batchloader(size, batchsize=100, g=basegrammar, N=5, compute_sketches=False, dc_model=None, shuffle=True, top_k_sketches=20, inv_temp=1.0, reward_fn=None, sample_fn=None, use_timeout=False):
	if batchsize==1:
		data = (sample_datum(g=g, N=N, compute_sketches=compute_sketches, dc_model=dc_model, top_k_sketches=20, inv_temp=inv_temp, reward_fn=reward_fn, sample_fn=sample_fn, use_timeout=use_timeout) for _ in repeat(0))
		yield from islice((x for x in data if x is not None), size)
	else:
		data = (sample_datum(g=g, N=N, compute_sketches=compute_sketches, dc_model=dc_model, top_k_sketches=20, inv_temp=inv_temp, reward_fn=reward_fn, sample_fn=sample_fn, use_timeout=use_timeout) for _ in repeat(0))
		data = (x for x in data if x is not None)
		grouped_data = islice(grouper(data, batchsize), size)

		for group in grouped_data:
			tps, ps, pseqs, IOs, sketchs, sketchseqs, rewards, sketchprobs = zip(*[(datum.tp, datum.p, datum.pseq, datum.IO, datum.sketch, datum.sketchseq, datum.reward, datum.sketchprob) for datum in group if datum is not None])
			yield Batch(tps, ps, pseqs, IOs, sketchs, sketchseqs, torch.FloatTensor(rewards) if any(r is not None for r in rewards) else None, torch.FloatTensor(sketchprobs) if any(s is not None for s in sketchprobs) else None)  # check that his works 

#TODO
def makeTestdata(synth=True, challenge=False):
	raise NotImplementedError
	tasks = []
	if synth:
		tasks = makeTasks()	
	if challenge:
		challenge_tasks, _ = loadPBETasks()
		tasks = tasks + challenge_tasks

	tasklist = []
	for task in tasks:
		if task.stringConstants==[] and task.request == arrow(tlist(tcharacter), tlist(tcharacter)):

				IO = tuple( (''.join(x[0]), ''.join(y)) for x,y in task.examples)

				program = None
				pseq = None
				sketch, sketchseq, reward, sketchprob = None, None, None, None
				tp = tprogram

				tasklist.append( Datum(tp, program, pseq, IO, sketch, sketchseq, reward, sketchprob) )

	return tasklist

#TODO
def loadTestTasks(path='rb_test_tasks.p'):
	raise NotImplementedError
	print("data file:", path)
	with open(path, 'rb') as datafile:
		tasks = pickle.load(datafile)
	return tasks

if __name__=='__main__':

	import time
	import pregex as pre
	
	g = basegrammar
	d = sample_datum(g=g, N=4, compute_sketches=True, top_k_sketches=100, inv_temp=1.0, reward_fn=None, sample_fn=None, dc_model=None)
	print(d.p)
	print(d.p.evaluate([]))
	print(d.sketch)
	#print(d.sketch.evaluate([])(pre.String("")))
	print(d.sketch.evaluate([]))
	print(d.sketchseq)
	for o in d.IO:
		print("example")
		print(o)
	

	from util.regex_util import PregHole, pre_to_prog


	preg = pre.create(d.sketchseq, lookup={PregHole:PregHole()})
	print(preg)
	print(pre_to_prog(preg))

