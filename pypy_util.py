#pypy_util.py

import sys
sys.path.append("/om/user/mnye/ec")
import time
from program import ParseFailure, Context
from grammar import NoCandidates, Grammar
from utilities import timing, callCompiled
from collections import namedtuple
from itertools import islice, zip_longest
from functools import reduce

DeepcoderResult = namedtuple("DeepcoderResult", ["sketch", "prog", "hit", "n_checked", "time"])

def test_program_on_IO(e, IO):
	return all(reduce(lambda a, b: a(b), xs, e)==y for xs, y in IO)

def alternate(*args):
	# note: python 2 - use izip_longest
	for iterable in zip_longest(*args):
		for item in iterable:
			if item is not None:
				yield item


def dc_enumerate(g, tp, IO, mdl, sketches, n_checked, n_hit, t, max_to_check):
	results = []
	for sk, x in alternate(*(((sk, x) for x in g.sketchEnumeration(Context.EMPTY, [], tp, sk, mdl)) for sk in sketches)):
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


def pypy_enumerate(g, tp, IO, mdl, sketches, n_checked, n_hit, t, max_to_check):
	# import copy
	# g = copy.deepcopy(g)
	# tp = copy.copy(tp)
	# IO = copy.copy(IO)
	# mdl = copy.copy(mdl)
	# sketches = copy.deepcopy(sketches)
	# n_checked = copy.deepcopy(n_checked)
	# n_hit = copy.copy(n_hit)
	return callCompiled(dc_enumerate, g, tp, IO, mdl, sketches, n_checked, n_hit, t, max_to_check)