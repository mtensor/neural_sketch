#rb_pypy_util.py

import sys
sys.path.append("/om/user/mnye/ec")
import time
from program import ParseFailure, Context
from grammar import NoCandidates, Grammar
from utilities import timing, callCompiled
from collections import namedtuple
from itertools import islice, zip_longest
from functools import reduce
from RobustFillPrimitives import robustFillPrimitives
from pypy_util import alternate

#A nasty hack!!!
max_len = 25
max_index = 4

_ = robustFillPrimitives(max_len=max_len, max_index=max_index)

RobustFillResult = namedtuple("RobustFillResult", ["sketch", "prog", "hit", "n_checked", "time", "g_hit"])

def test_program_on_IO(e, IO, n_rec_examples, generalization=False):
	examples = IO if generalization else IO[:n_rec_examples]
	try: 
		return all(e(x)==y for x, y in examples)  #TODO: check that this makes sense
	except IndexError:
		return False

def rb_enumerate(g, tp, IO, mdl, sketches, n_checked, n_hit, t, max_to_check, test_generalization, n_rec_examples):
	results = []
	for sk, x in alternate(*(((sk, x) for x in g.sketchEnumeration(Context.EMPTY, [], tp, sk, mdl)) for sk in sketches)):
		_, _, p = x
		e = p.evaluate([])
		try:
			hit = test_program_on_IO(e, IO, n_rec_examples)
		except: hit = False
		prog = p if hit else None
		n_checked += 1
		n_hit += 1 if hit else 0
		#testing generalization
		gen_hit = test_program_on_IO(e, IO, n_rec_examples, generalization=True) if test_generalization else False
		results.append( RobustFillResult(sk, prog, hit, n_checked, time.time()-t, gen_hit))
		if hit: break
		if n_checked >= max_to_check: break
	return results, n_checked, n_hit


def rb_pypy_enumerate(g, tp, IO, mdl, sketches, n_checked, n_hit, t, max_to_check, test_generalization, n_rec_examples):
	return callCompiled(rb_enumerate, g, tp, IO, mdl, sketches, n_checked, n_hit, t, max_to_check, test_generalization, n_rec_examples)