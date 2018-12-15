#generate deepcoder data

import pickle
# TODO
from util.deepcoder_util import make_holey_deepcoder # this might be enough
#from util.robustfill_util import basegrammar # TODO
from util.robustfill_util import sample_program, generate_IO_examples, timing # TODO

import time
from collections import namedtuple
#Function = namedtuple('Function', ['src', 'sig', 'fun', 'bounds'])

from grammar import Grammar, NoCandidates
from utilities import flatten

# TODO
from RobustFillPrimitives import RobustFillProductions, flatten_program, tprogram

from program import Application, Hole, Primitive, Index, Abstraction, ParseFailure
import math
import random
from type import Context, arrow, tint, tlist, UnificationFailure, tcharacter
from itertools import zip_longest, chain, repeat, islice
from functools import reduce
import torch
from makeTextTasks import makeTasks, loadPBETasks
from util.algolisp_util import make_holey_algolisp


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


def sample_datum(basegrammar,
					N=5,
					V=100,
					L=10,
					compute_sketches=False,
					top_k_sketches=100,
					inv_temp=1.0,
					reward_fn=None,
					sample_fn=None,
					dc_model=None,
					use_timeout=False,
					improved_dc_model=False,
					nHoles=1):

	#sample a program:
	#with timing("sample program"):
	program = sample_program(g=basegrammar, max_len=L, max_string_size=V)  # TODO
	# if program is bad: return None  # TODO

	# find IO
	#with timing("sample IO:"):
	IO = generate_IO_examples(program, num_examples=N,  max_string_size=V)  # TODO
	if IO is None: return None
	IO = tuple(IO)
	# find tp
	tp = tprogram
	# TODO

	# find pseq
	pseq = tuple(flatten_program(program)) #TODO

	if compute_sketches:
		# find sketch
		#grammar = g if not dc_model else dc_model.infer_grammar(IO)
		#with timing("make_holey"):
		# sketch, reward, sketchprob = make_holey_deepcoder(program,
		# 													top_k_sketches,
		# 													grammar,
		# 													tp,
		# 													inv_temp=inv_temp,
		# 													reward_fn=reward_fn,
		# 													sample_fn=sample_fn,
		# 													use_timeout=use_timeout) #TODO

		sketch, reward, sketchprob = make_holey_algolisp(program,
													top_k_sketches,
													tp,
													basegrammar,
													dcModel=dc_model,
													improved_dc_model=improved_dc_model,
													return_obj=Hole,
													dc_input=IO,
													inv_temp=inv_temp,
													reward_fn=reward_fn,
													sample_fn=sample_fn,
													use_timeout=use_timeout,
													nHoles=nHoles,
													domain='text')

		# find sketchseq
		sketchseq = tuple(flatten_program(sketch))
	else:
		sketch, sketchseq, reward, sketchprob = None, None, None, None

	return Datum(tp, program, pseq, IO, sketch, sketchseq, reward, sketchprob)


def grouper(iterable, n, fillvalue=None):
	"Collect data into fixed-length chunks or blocks"
	# grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
	args = [iter(iterable)] * n
	return zip_longest(*args, fillvalue=fillvalue)


# max_iteration - dcModel.iteration,
#                                                 batchsize=1,
#                                                 g=basegrammar,
#                                                 N=args.n_examples,
#                                                 V=args.max_length,
#                                                 L=args.max_list_length, 
#                                                 compute_sketches=args.improved_dc_model,
#                                                 dc_model=dcModel if use_dc_grammar and (dcModel.epochs > 1) else None, # TODO
#                                                 improved_dc_model=args.improved_dc_model,
#                                                 top_k_sketches=args.k,
#                                                 inv_temp=args.inv_temp,
#                                                 nHoles=args.nHoles,
#                                                 use_timeout=args.use_timeout

def batchloader(size,
				basegrammar,
				batchsize=100,
				N=5,
				V=100,
				L=10,
				compute_sketches=False,
				dc_model=None,
				shuffle=True, 
				top_k_sketches=20,
				inv_temp=1.0,
				reward_fn=None,
				sample_fn=None,
				use_timeout=False,
				improved_dc_model=False,
				nHoles=1):
	data = (sample_datum(basegrammar,
							N=N,
							V=V,
							L=L,
							compute_sketches=compute_sketches,
							dc_model=dc_model,
							top_k_sketches=20,
							inv_temp=inv_temp,
							reward_fn=reward_fn,
							sample_fn=sample_fn,
							use_timeout=use_timeout,
							improved_dc_model=improved_dc_model,
							nHoles=nHoles) for _ in repeat(0))
	data = (x for x in data if x is not None)
	if batchsize==1:	
		yield from islice(data, size)
	else:
		grouped_data = islice(grouper(data, batchsize), size)
		for group in grouped_data:
			tps, ps, pseqs, IOs, sketchs, sketchseqs, rewards, sketchprobs = zip(*[(datum.tp, datum.p, datum.pseq, datum.IO, datum.sketch, datum.sketchseq, datum.reward, datum.sketchprob) for datum in group if datum is not None])
			yield Batch(tps, ps, pseqs, IOs, sketchs, sketchseqs, torch.FloatTensor(rewards) if any(r is not None for r in rewards) else None, torch.FloatTensor(sketchprobs) if any(s is not None for s in sketchprobs) else None)  # check that his works 


def makeTestdata(synth=True, challenge=False):
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


# tasks = makeTestdata(synth=True, challenge=True)
# with open('rb_all_tasks.p', 'wb') as savefile:
# 	pickle.dump(tasks, savefile)
# 	print('saved rb challenge tasks')


def loadTestTasks(path='rb_test_tasks.p'):
	print("data file:", path)
	with open(path, 'rb') as datafile:
		tasks = pickle.load(datafile)
	return tasks

if __name__=='__main__':
	import time
	
	g = Grammar.fromProductions(RobustFillProductions(max_len=50, max_index=4))
	d = sample_datum(g=g, N=4, V=50, L=10, compute_sketches=True, top_k_sketches=100, inv_temp=1.0, reward_fn=None, sample_fn=None, dc_model=None)
	print(d.p)
	for i,o in d.IO:
		print("example")
		print(i)
		print(o)

	tasks = loadTestTasks('rb_all_tasks.p')
	for t in tasks: print(t.IO)

	#loader = batchloader(600, g=g, batchsize=200, N=5, V=50, L=10, compute_sketches=True, dc_model=None, shuffle=True, top_k_sketches=10)

	# t = time.time()
	# for batch in loader:
	# 	print(time.time() - t)
	# 	print(batch.IOs[0])
	# 	print(batch.ps[0])

	# print(d)
	# if d is not None:
	# 	print(d.p)
	# 	print(d.IO)
	# 	print(d.sketch)
	# from itertools import islice
	# convert_source_to_datum("a <- [int] | b <- [int] | c <- ZIPWITH + b a | d <- COUNT isEVEN c | e <- ZIPWITH MAX a c | f <- MAP MUL4 e | g <- TAKE d f")

	# filename = 'data/DeepCoder_data/T2_A2_V512_L10_train_perm.txt'
	# train_data = 'data/DeepCoder_data/T3_A2_V512_L10_train_perm.txt'

	# test_data = ''

	# lines = (line.rstrip('\n') for i, line in enumerate(open(filename)) if i != 0) #remove first line

	# for datum in islice(batchloader([train_data], batchsize=1, N=5, V=128, L=10, compute_sketches=True, top_k_sketches=20, inv_temp=0.05), 30):
	# 	print("program:", datum.p)
	# 	print("sketch: ", datum.sketch)
		
	#path = 'data/pretrain_data_v1_alt.p'
	#make_deepcoder_data(path, with_holes=True, k=20)