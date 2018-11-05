#regex_pypy_util.py


import time
from program import ParseFailure, Context
from grammar import NoCandidates, Grammar
from utilities import timing, callCompiled
from collections import namedtuple
from itertools import islice, zip_longest
from functools import reduce
import pregex as pre #oh god

RegexResult = namedtuple("RegexResult", ["sketch", "prog", "ll", "n_checked", "time"])
SketchTup = namedtuple("SketchTup", ['sketch', 'g'])


def alternate(*args):
	# note: python 2 - use izip_longest
	for iterable in zip_longest(*args):
		for item in iterable:
			if item is not None:
				yield item

def regex_enumerate(tp, IO, mdl, sketchtups, n_checked, n_hit, t, max_to_check):
	results = []
	best_ll = float('-inf')
	for sketch, x in alternate(*(((sk.sketch, x) for x in sk.g.sketchEnumeration(Context.EMPTY, [], tp, sk.sketch, mdl)) for sk in sketchtups)): #TODO!! sketches n grammars n stuff
		_, _, p = x
		e = p.evaluate([])
		ll = sum(e.match(example)/len(example) for example in IO[nExamples:])/len(IO[nExamples:])  # TODO -- check this
		if ll > best_ll:
			best_ll = ll
		hit = ( ll is not float('-inf') ) 
		prog = p if hit else None
		n_checked += 1
		n_hit += 1 if hit else 0
		results.append((RegexResult(sketch, prog, ll, n_checked, time.time()-t), best_ll))
		if n_checked >= max_to_check: break

	return results, n_checked, n_hit

def pypy_enumerate(tp, IO, mdl, sketchtups, n_checked, n_hit, t, max_to_check):
	return callCompiled(regex_enumerate, tp, IO, mdl, sketchtups, n_checked, n_hit, t, max_to_check)