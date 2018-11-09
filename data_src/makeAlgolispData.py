#generate deepcoder data
import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./ec'))

import pickle
import time
from collections import namedtuple
#Function = namedtuple('Function', ['src', 'sig', 'fun', 'bounds'])

from grammar import Grammar, NoCandidates
from utilities import flatten

from algolispPrimitives import tsymbol, algolispProductions, algolispPrimitives
from program import Application, Hole, Primitive, Index, Abstraction, ParseFailure
import math
import random
from type import Context, arrow, tint, tlist, UnificationFailure

from util.algolisp_util import convert_IO, tree_to_prog, make_holey_algolisp, AlgolispHole, tree_to_seq

from itertools import zip_longest, chain
from functools import reduce
import torch

from program_synthesis.algolisp.dataset import dataset
from program_synthesis.algolisp import arguments


#I want a list of namedTuples

#Datum = namedtuple('Datum', ['tp', 'p', 'pseq', 'IO', 'sketch', 'sketchseq'])

basegrammar = Grammar.fromProductions(algolispProductions())

#redo this ...
class Datum():
	def __init__(self, tp, p, pseq, IO, sketch, sketchseq, reward, sketchprob, spec, schema_args):
		self.tp = tp
		self.p = p
		self.pseq = pseq
		self.IO = IO
		self.sketch = sketch
		self.sketchseq = sketchseq
		self.reward = reward
		self.sketchprob = sketchprob
		self.spec = spec
		self.schema_args = schema_args

	def __hash__(self): 
		return reduce(lambda a, b: hash(a + hash(b)), flatten(self.spec, abort=lambda x: type(x) is str), 0) + hash(self.p) + hash(self.sketch)

Batch = namedtuple('Batch', ['tps', 'ps', 'pseqs', 'IOs', 'sketchs', 'sketchseqs', 'rewards', 'sketchprobs','specs', 'schema_args'])

def convert_datum(ex, compute_sketches=False, top_k_sketches=20, inv_temp=1.0, reward_fn=None, sample_fn=None, dc_model=None, improved_dc_model=True, use_timeout=False, proper_type=False):
	"""
	for ex in b:
		ex.text
		ex.tests
		ex.code_tree
	"""

	#find IO
	IO = convert_IO(ex.tests) #TODO

	schema_args = ex.schema.args

	# find tp
	if proper_type:
		assert False #arrow(some_stuff, tsymbol)
		out_tp = convert_tp(x.schema.return_type)
	else:
		tp = tsymbol

	# find program p
	pseq = tuple(ex.code_sequence)

	# find pseq
	if proper_type:
		assert False
	else:
		p = tree_to_prog(ex.code_tree)  # TODO: use correct grammar, and 

	spec = ex.text

	if compute_sketches:
		# find sketch
		#grammar = basegrammar if not dc_model else dc_model.infer_grammar(IO) #This line needs to change
		dc_input = spec
		sketch, reward, sketchprob = make_holey_algolisp(p, top_k_sketches, tp, basegrammar=basegrammar, dcModel=dc_model, improved_dc_model=improved_dc_model, inv_temp=inv_temp, reward_fn=reward_fn, sample_fn=sample_fn, use_timeout=use_timeout, return_obj=AlgolispHole, dc_input=dc_input) #TODO

		# find sketchseq
		sketchseq = tuple(tree_to_seq(sketch.evaluate([])))
	else:
		sketch, sketchseq, reward, sketchprob = None, None, None, None

	return Datum(tp, p, pseq, IO, sketch, sketchseq, reward, sketchprob, spec, schema_args)


def batchloader(data_file, batchsize=100, compute_sketches=False, dc_model=None, improved_dc_model=True, shuffle=True, top_k_sketches=20, inv_temp=1.0, reward_fn=None, sample_fn=None, use_timeout=False):

	parser = arguments.get_arg_parser('Training AlgoLisp', 'train')
	args = parser.parse_args()
	args.cuda = not args.no_cuda and torch.cuda.is_available()

	args.batch_size = batchsize # ARGS
	print("args.batch_size", args.batch_size)

	if data_file == 'train':
		data, _ = dataset.get_dataset(args)
	elif data_file == 'dev':
		_, data = dataset.get_dataset(args)
	elif data_file == 'eval':
		data = dataset.get_eval_dataset(args)
	else:
		assert False

	#figure out how to deal with these
	if batchsize==1:
		data = (convert_datum(ex,
			compute_sketches=compute_sketches,
			top_k_sketches=top_k_sketches,
			inv_temp=inv_temp,
			reward_fn=reward_fn,
			sample_fn=sample_fn,
			dc_model=dc_model,
			improved_dc_model=improved_dc_model,
			use_timeout=use_timeout,
			proper_type=False) for batch in data for ex in batch) #I assume batch has one ex
		yield from (x for x in data if x is not None)

	else:
		#then i want
		for batch in data:
			tps, ps, pseqs, IOs, sketchs, sketchseqs, rewards, sketchprobs, specs, schema_args = zip(*[
				(datum.tp, datum.p, datum.pseq, datum.IO, datum.sketch, datum.sketchseq, datum.reward, datum.sketchprob, datum.spec, datum.schema_args)
				for datum in (convert_datum(ex,
									compute_sketches=compute_sketches,
									top_k_sketches=top_k_sketches,
									inv_temp=inv_temp,
									reward_fn=reward_fn,
									sample_fn=sample_fn,
									dc_model=dc_model,
									improved_dc_model=improved_dc_model,
									use_timeout=use_timeout,
									proper_type=False) for ex in batch)
				 					if datum is not None])
			
			yield Batch(tps, ps, pseqs, IOs, sketchs, sketchseqs, torch.FloatTensor(rewards) if any(r is not None for r in rewards) else None, torch.FloatTensor(sketchprobs) if any(s is not None for s in sketchprobs) else None, specs, schema_args)  # check that his work

if __name__=='__main__':
	from itertools import islice


	algolispProductions()

	d = islice(batchloader('train', batchsize=200, compute_sketches=True, dc_model=None, improved_dc_model=True, shuffle=True, top_k_sketches=20, inv_temp=1.0, reward_fn=None, sample_fn=None, use_timeout=False),100)

	for datum in d:
		print("program:", datum.p)
		print("sketch: ", datum.sketch)
		print(len(datum.pseq))
		print()






