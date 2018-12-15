#pypy_util.py


import time
from program import ParseFailure, Context
from grammar import NoCandidates, Grammar
from utilities import timing, callCompiled
from collections import namedtuple
from itertools import islice, zip_longest
from functools import reduce
from deepcoderPrimitives import deepcoderPrimitives

SketchTup = namedtuple("SketchTup", ['sketch', 'g'])
DeepcoderResult = namedtuple("DeepcoderResult", ["sketch", "prog", "hit", "n_checked", "time"])
deepcoderPrimitives()

def test_program_on_IO(e, IO):
	return all(reduce(lambda a, b: a(b), xs, e)==y for xs, y in IO)

def alternate(*args):
	# note: python 2 - use izip_longest
	for iterable in zip_longest(*args):
		for item in iterable:
			if item is not None:
				yield item


def dc_enumerate(tp, IO, mdl, sketchtups, n_checked, n_hit, t, max_to_check):
	results = []

	f = lambda tup: map( lambda x: (tup.sketch, x), tup.g.sketchEnumeration(Context.EMPTY, [], tp, tup.sketch, mdl, maximumDepth=20))
	sIterable = list(map(f, sketchtups))


	for sk, x in alternate(*sIterable):
		_, _, p = x
		e = p.evaluate([])
		hit = test_program_on_IO(e, IO)
		prog = p if hit else None
		n_checked += 1
		n_hit += 1 if hit else 0
		results.append( DeepcoderResult(sk, prog, hit, n_checked, time.time()-t) )
		if hit: break
		if n_checked >= max_to_check: break

	return results, n_checked, n_hit


def pypy_enumerate(g, tp, IO, mdl, sketchtups, n_checked, n_hit, t, max_to_check):
	return callCompiled(dc_enumerate, tp, IO, mdl, sketchtups, n_checked, n_hit, t, max_to_check)