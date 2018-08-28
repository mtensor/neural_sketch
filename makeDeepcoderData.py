#generate deepcoder data

import pickle

from deepcoder_util import parseprogram, make_holey_deepcoder
import time

from collections import namedtuple
#Function = namedtuple('Function', ['src', 'sig', 'fun', 'bounds'])

import sys
sys.path.append("/om/user/mnye/ec")

from grammar import Grammar, NoCandidates
from deepcoderPrimitives import deepcoderProductions, flatten_program

from program import Application, Hole, Primitive, Index, Abstraction, ParseFailure

import math
from type import Context, arrow, tint, tlist, UnificationFailure

from dc_program import generate_IO_examples, compile

from itertools import zip_longest

#from dc_program import Program as dc_Program

def make_deepcoder_data(filename, with_holes=False, size=10000000, k=20):
	data_list = []
	save_freq = 1000

	t = time.time()
	for i in range(size):
		inst = getInstance(with_holes=with_holes, k=k)
		data_list.append(inst)


		if i%save_freq==0:
			#save data
			
			print(f"iteration {i} out of {size}")
			print("saving data")
			with open(filename, 'wb') as file:
				pickle.dump(data_list, file)
			print(f"time since last {save_freq}: {time.time()-t}")
			t = time.time()



#I want a list of namedTuples

Datum = namedtuple('Datum', ['tp', 'p', 'pseq', 'IO', 'sketch', 'sketchseq'])
Batch = namedtuple('Batch', ['tps', 'ps', 'pseqs', 'IOs', 'sketchs', 'sketchseqs'])

def convert_dc_program_to_ec(dc_program, tp):
	source = dc_program.src
	source = source.split('\n')
	source = [line.split(' ') for line in source]
	#print(source)
	
	num_inputs = len(tp.functionArguments())
	#print(num_inputs)

	del source[:num_inputs]
	source = [[l for l in line if l != '<-'] for line in source]

	#print(source)

	last_var = source[-1][0]
	prog = source[-1][1:]
	del source[-1]

	variables = list('abcdefghigklmnop')
	del variables[variables.index(last_var):] #check this line

	#print(variables)

	lookup = {variables[i]: ["input_" + str(i)] for i in range(num_inputs)}
	#del variables[:num_inputs]

	for line in source:
		lookup[line[0]] = line[1:]

	for variable in reversed(variables):
		p2 = []
		for x in prog:
			if x==variable:
				p2 += lookup[variable]
			else:
				p2.append(x)

		prog = p2
		#prog = [ lookup[variable] if x==variable else [x] for x in prog]
	return prog


def convert_source_to_datum(source, N=5, V=512, L=10, compute_sketches=False):
	source = source.replace(' | ', '\n')
	dc_program = compile(source, V=V, L=L)

	#find IO
	IO = generate_IO_examples(dc_program, N=N, L=L, V=V)

	#find tp
	ins = [tint if inp == int else tlist(tint) for inp in dc_program.ins] 
	if dc_program.out == int:
		out = tint
	else:
		assert dc_program.out==[int]
		out = tlist(tint)
	tp = arrow( *(ins+[out]) )

	#find program p
	pseq = convert_dc_program_to_ec(dc_program, tp)

	#find pseq
	p = parseprogram(pseq, tp) #TODO: use correct grammar, and 

	if compute_sketches:
		#find sketch
		k = 20
		sketch = make_holey_deepcoder(p, k, grammar, tp) #TODO

		#find sketchseq
		sketchseq = flatten_program(sketch)
	else:
		sketch, sketchseq = None, None

	return Datum(tp, p, pseq, IO, sketch, sketchseq)


def grouper(iterable, n, fillvalue=None):
	"Collect data into fixed-length chunks or blocks"
	# grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
	args = [iter(iterable)] * n
	return zip_longest(*args, fillvalue=fillvalue)


def batchloader(lines, batchsize=100, N=5, V=512, L=10, compute_sketches=False):
        
	grouped_lines = grouper(lines, batchsize)

	for group in grouped_lines:
		data = (convert_source_to_datum(line, N=N, V=V, L=L, compute_sketches=compute_sketches) for line in group)

		tps, ps, pseqs, IOs, sketchs, sketchseqs = zip(*[(datum.tp, datum.p, datum.pseq, datum.IO, datum.sketch, datum.sketchseq) for datum in data])

		yield Batch(tps, ps, pseqs, IOs, sketchs, sketchseqs)




if __name__=='__main__':


	convert_source_to_datum("a <- [int] | b <- [int] | c <- ZIPWITH + b a | d <- COUNT isEVEN c | e <- ZIPWITH MAX a c | f <- MAP MUL4 e | g <- TAKE d f")

	filename = 'data/DeepCoder_data/T2_A2_V512_L10_train_perm.txt'
	train_data = 'data/DeepCoder_data/T3_A2_V512_L10_train_perm.txt'

	test_data = ''

	lines = (line.rstrip('\n') for i, line in enumerate(open(filename)) if i != 0) #remove first line

	for batch in batchloader(lines):
		assert False





	#path = 'data/pretrain_data_v1_alt.p'
	#make_deepcoder_data(path, with_holes=True, k=20)




