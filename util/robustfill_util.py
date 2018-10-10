#deepcoder_util.py
#import sys
#import os
#sys.path.append(os.path.abspath('./'))

from builtins import super
import pickle
import string
import argparse
import random

import torch
from torch import nn, optim

from pinn import RobustFill
from pinn import SyntaxCheckingRobustFill
import random
import math
import time
from string import printable

from pregex import pregex as pre

from collections import OrderedDict
#from util import enumerate_reg, Hole

import re


from grammar import Grammar, NoCandidates
#from deepcoderPrimitives import deepcoderProductions, flatten_program
from RobustFillPrimitives import RobustFillProductions, flatten_program, tprogram, Constraint_prop, delimiters
from program import Application, Hole, Primitive, Index, Abstraction, ParseFailure, prettyProgram
import math
from type import Context, arrow, UnificationFailure

productions = RobustFillProductions()  # TODO - figure out good production probs ... 
basegrammar = Grammar.fromProductions(productions, logVariable=0.0)  # TODO

def robustfill_vocab(grammar): 
    return [prim.name for prim in grammar.primitives] + ['<HOLE>']  # TODO

class timing(object):
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        print(self.message, f"in {time.time() - self.start} seconds")

d = {re.escape(i):i for i in delimiters}
d['\\.'] = '\\.'
d['\\)'] = '\\)'
d['\\('] = '\\('
d['('] = '\\('
d[')'] = '\\)'
preg_dict = {r'[A-Z][a-z]+': '\\u\\l+', r'[A-Z]': '\\u', r'[a-z]': '\\l', **d}  #note: uses escaped versions of delimiters for constraints 

def extract_constraints(program):  # TODO
    #throw an issue if min bigger than max
    return Constraint_prop().execute(program)

def sample_program(g=basegrammar, max_len=10, max_string_size=100):
    request = tprogram
    #with timing("sample from grammar"):
    p = g.sample(request, maximumDepth=5, maxAttempts=None)  #todo args??

    if flatten_program(p).count('concat_list') >= max_len:
        #resample
        return sample_program(g=g, max_len=max_len, max_string_size=max_string_size)  # TODO
    else: return p


def generate_inputs_from_constraints(constraint_dict, min_size, max_string_size=100):
    #sample a size from min to max
    size = random.randint(min_size, max_string_size)
    indices = set(range(size))
    slist = random.choices(printable[:-4] , k=size)
    # schematically:
    #print("min_size", min_size)
    #print("size", size)
    for item in constraint_dict:
        #print("ITEM", item)
        #print("sliststr:", ''.join(slist))
        num_to_insert = max(0, constraint_dict[item] - len(re.findall(re.escape(item), ''.join(slist))))
        #print("num_to_insert", num_to_insert)
        if len(indices) < num_to_insert: return None
        indices_to_insert = set(random.sample(indices, k=num_to_insert))
        # do something here
        #print("PREG INPUT",item if item not in preg_dict else preg_dict[item])
        for i in indices_to_insert:
            slist[i] = pre.create(item).sample() if item not in preg_dict else pre.create(preg_dict[item]).sample()
        indices = indices - indices_to_insert
        #may be too big but whatever
    string = ''.join(slist)
    if len(string) > max_string_size: return string[:max_string_size] # may break but whatever 
    return string

def generate_IO_examples(program, num_examples=5,  max_string_size=100):
    constraint_dict, min_size = extract_constraints(program) # check here that 
    if min_size > max_string_size: return None
    """
    need to generate num_examples examples. 
    can go wrong if:
        min_size > max_string_size - sample new prog - do that in extract_constraints and sample new prog
        program(input) > max_string_size - can do a few tries 
        program(input) throws an error bc constraint prop wasn't perfect - can do a few tries 
    """
    examples = []
    for _ in range(2*num_examples):
        #with timing("generate_from constraints"):
        instring = generate_inputs_from_constraints(constraint_dict, min_size, max_string_size=max_string_size)
        if instring is None: continue
        try: outstring = program.evaluate([])(instring)
        except IndexError: continue
        if len(outstring) > max_string_size: continue  # might cheange to return None for speed
        examples.append((instring, outstring))
        if len(examples) >= num_examples: break
    else: return None
    return examples



def tokenize_for_robustfill(IOs):
    """
    tokenizes a batch of IOs ... I think none is necessary ...
    """
    return IOs

def buildCandidate(request, context, environment, parsecontext): #TODO
    """Primitives that are candidates for being used given a requested type
    If returnTable is false (default):
    returns [((log)likelihood, tp, primitive, context)]
    if returntable is true: returns {primitive: ((log)likelihood, tp, context)}"""
    variable_list = ['input_' + str(i) for i in range(4)]

    if len(parsecontext) == 0: raise NoCandidates()
    chosen_str = parsecontext[0]
    parsecontext = parsecontext[1:] #is this right?

    candidate = None
    #for l, t, p in self.productions:
    if chosen_str in Primitive.GLOBALS: #if it is a primtive
        p = Primitive.GLOBALS[chosen_str]
        t = p.tp 
        try:
            newContext, t = t.instantiate(context)
            newContext = newContext.unify(t.returns(), request)
            t = t.apply(newContext)
            #candidates.append((l, t, p, newContext))
            candidate = (t, p, newContext)

        except UnificationFailure:
            raise ParseFailure()

    else: #if it is a hole:
        try: assert chosen_str == '<HOLE>' #TODO, choose correct representation of program
        except AssertionError as e:
            print("bad string:", chosen_str)
            assert False
        p = Hole()
        t = request #[try all possibilities and backtrack] #p.inferType(context, environment, freeVariables) #TODO
        # or hole is request.
        try:
            newContext, t = t.instantiate(context)
            newContext = newContext.unify(t.returns(), request)
            t = t.apply(newContext)
            #candidates.append((l, t, p, newContext))
            candidate = (t, p, newContext)

        except UnificationFailure:
            raise ParseFailure()
    if candidate == None:
        raise NoCandidates()
    return parsecontext, candidate


def parseprogram(pseq, request): #TODO 
    num_inputs = len(request.functionArguments())
    #request = something #TODO
    def _parse(request, parsecontext, context, environment):
        if request.isArrow():
            parsecontext, context, expression = _parse(
                request.arguments[1], parsecontext, context, [
                    request.arguments[0]] + environment)
            return parsecontext, context, Abstraction(expression) #TODO

        parsecontext, candidate = buildCandidate(request, context, environment, parsecontext)
        newType, chosenPrimitive, context = candidate
   
        # Sample the arguments
        xs = newType.functionArguments()
        returnValue = chosenPrimitive
        for x in xs:
            x = x.apply(context)
            parsecontext, context, x = _parse(
                x, parsecontext, context, environment)
            returnValue = Application(returnValue, x)
        return parsecontext, context, returnValue
    _, _, e = _parse(
                    request, pseq, Context.EMPTY, [])
    return e

def make_holey_deepcoder(prog, k, g, request, inv_temp=1.0, reward_fn=None, sample_fn=None):  # TODO
    """
    inv_temp==1 => use true mdls
    inv_temp==0 => sample uniformly
    0 < inv_temp < 1 ==> something in between
    """ 
    choices = g.enumerateHoles(request, prog, k=k)
    if len(list(choices)) == 0:
        #if there are none, then use the original program 
        choices = [(prog, 0)]
    #print("prog:", prog, "choices", list(choices))
    progs, weights = zip(*choices)
    #normalize weights, and then rezip
    if reward_fn is None:
        reward_fn = math.exp
    if sample_fn is None:
        sample_fn = lambda x: math.exp(inv_temp*x)
    rewards = [reward_fn(w) for w in weights]
    weights = [sample_fn(w) for w in weights]
    #normalize_weights
    w_sum = sum(w for w in weights)
    weights = [w/w_sum for w in weights]
    
    prog_reward_probs = list(zip(progs, rewards, weights))

    if k > 1:
        x = random.choices(prog_reward_probs, weights=weights, k=1)
        return x[0] #outputs prog, prob
    else:
        return prog_reward_probs[0] #outputs prog, prob

if __name__=='__main__':
    g = Grammar.fromProductions(RobustFillProductions())
    print(len(g))
    request = tprogram
    p = g.sample(request)
    print("request:", request)
    print("program:")
    print(prettyProgram(p))
    s = 'abcdefg'
    e = p.evaluate([])
    print("prog applied to", s)
    print(e(s))
    print("flattened_program:")
    flat = flatten_program(p)
    print(flat)
    pr = parseprogram(flat, request)
    print(prettyProgram(pr))

