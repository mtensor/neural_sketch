#pypy_util.py


import time
from program import ParseFailure, Context
from grammar import NoCandidates, Grammar, SketchEnumerationFailure
from utilities import timing, callCompiled
from collections import namedtuple
from itertools import islice, zip_longest
from functools import reduce

from program_synthesis.algolisp.dataset import executor
#from memory_profiler import profile

SketchTup = namedtuple("SketchTup", ['sketch', 'g'])
AlgolispResult = namedtuple("AlgolispResult", ["sketch", "prog", "hit", "n_checked", "time"])


#from algolisp code
#executor_ = executor.LispExecutor()

# #for reference:
# def get_stats_from_code(args):
#     res, example, executor_ = args
#     if len(example.tests) == 0:
#         return None
#     if executor_ is not None:
#         stats = executor.evaluate_code(
#             res.code_tree if res.code_tree else res.code_sequence, example.schema.args, example.tests,
#             executor_)
#         stats['exact-code-match'] = is_same_code(example, res)
#         stats['correct-program'] = int(stats['tests-executed'] == stats['tests-passed'])
#     else: assert False
# #what is a res?


def test_program_on_IO(e, IO, schema_args, executor_):
	"""
	run executor
	"""
	stats = executor.evaluate_code(
		e, schema_args, IO,
		executor_)
	#print(stats['tests-executed'], stats['tests-passed'])
	return stats['tests-executed'] == stats['tests-passed']

def alternate(*args):
	# note: python 2 - use izip_longest
	for iterable in zip_longest(*args):
		for item in iterable:
			if item is not None:
				yield item


def algolisp_enumerate(tp, IO, schema_args, mdl, sketchtups, n_checked, n_hit, t, max_to_check):
	results = []
	executor_ = executor.LispExecutor()

	# empty = all(False for _ in alternate(*(((sk.sketch, x) for x in sk.g.sketchEnumeration(Context.EMPTY, [], tp, sk.sketch, mdl)) for sk in sketchtups)))
	# if empty:
	# 	print("ALTERNATE EMPTY") #TODO
		#print("lensketchups", len(sketchtups))
		#print(p for p in sk.g.sketchEnumeration(Context.EMPTY, [], tp, sk.sketch, mdl) for sk in sketchtups)

	#for sketch, xp in alternate(*(((sk.sketch, x) for x in sk.g.sketchEnumeration(Context.EMPTY, [], tp, sk.sketch, mdl)) for sk in sketchtups)):
	

	f = lambda tup: map( lambda x: (tup.sketch, x), tup.g.sketchEnumeration(Context.EMPTY, [], tp, tup.sketch, mdl, maximumDepth=100))
	sIterable = list(map(f, sketchtups))

	hit = False
	for sketch, xp in alternate(* sIterable ):
		_, _, p = xp
		e = p.evaluate([])
		#print(e)
		hit = test_program_on_IO(e, IO, schema_args, executor_)
		prog = p if hit else None
		n_checked += 1
		n_hit += 1 if hit else 0
		if hit: 
			results.append( AlgolispResult(sketch, prog, hit, n_checked, time.time()-t) )
			break
		if n_checked >= max_to_check:
			del sketch
			del xp
			break
	if n_checked < len(sketchtups) and not hit: print("WARNING: not all candidate sketches checked")

	del executor_
	del sIterable
	del f
	#print("ex cache len:")
	#print(len(executor.code_lisp._EXECUTION_CACHE))
	executor.code_lisp._EXECUTION_CACHE = {}
	return results, n_checked, n_hit


#pypy_enumerate(datum.tp, datum.IO, mdl, sketchtups, n_checked, n_hit, t, max_to_check)
def pypy_enumerate(tp, IO, schema_args, mdl, sketchtups, n_checked, n_hit, t, max_to_check):
	return callCompiled(algolisp_enumerate, tp, IO, schema_args, mdl, sketchtups, n_checked, n_hit, t, max_to_check)
	#if pypy doesn't work we can just call algolisp_enumerate normally
