#deepcoder_util.py
import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./ec'))

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

from collections import OrderedDict
#from util import enumerate_reg, Hole
from grammar import Grammar, NoCandidates
from program import Application, Hole, Primitive, Index, Abstraction, ParseFailure

import math
from type import Context, arrow, tint, tlist, tbool, UnificationFailure
from algolispPrimitives import tsymbol, tconstant, tfunction, primitive_lookup


from program_synthesis.algolisp.dataset import data
from program_synthesis.algolisp.dataset import executor
#primitive_lookup = {prim.name:prim.name for prim in napsPrimitives()}

def tokenize_for_robustfill(specs):
    return [[spec] for spec in specs]

def convert_IO(tests):
    return tests

def tree_to_seq(tree):
    return data.flatten_lisp_code(tree)  #from algolisp.data

def seq_to_tree(seq):
    #from algolisp code
    code, _ = data.unflatten_code(seq, 'lisp')
    return code

def tree_depth(tree):
    depth = 0
    for x in tree:
        if type(x)==list:
            depth = max(tree_depth(x), depth)
    return depth + 1

class AlgolispHole(Hole):
    def show(self, isFunction): return "<HOLE>"

    def evaluate(self, e):
        return '<HOLE>'

    @staticmethod
    def _parse(s,n):
        while n < len(s) and s[n].isspace(): n += 1
        n = Program.parseConstant(s,n,
                                  '<HOLE>')
        return AlgolispHole(), n

def make_holey_algolisp(prog, k, request, basegrammar, dcModel=None, improved_dc_model=False, return_obj=AlgolispHole, dc_input=None, inv_temp=1.0, reward_fn=None, sample_fn=None, verbose=False, use_timeout=False, nHoles=4):
    """
    inv_temp==1 => use true mdls
    inv_temp==0 => sample uniformly
    0 < inv_temp < 1 ==> something in between
    """ 
    if dcModel is None:
        #print("dcModel NONE")
        g = basegrammar
        choices = g.enumerateMultipleHoles(request, prog, k=k, return_obj=return_obj, nHoles=nHoles)
    elif dcModel and not improved_dc_model:
        g = dcModel.infer_grammar(dc_input) #(spec, sketch)
        choices = g.enumerateMultipleHoles(request, prog, k=k, return_obj=return_obj, nHoles=nHoles)
    else: 
        assert improved_dc_model
        g = basegrammar
        #print("time before hole punching", time.time())
        choices = g.enumerateMultipleHoles(request, prog, k=k, return_obj=return_obj, nHoles=nHoles) # request, full, sk
        #print("time after hole punching", time.time())
        # print("probs before grammar inf:")
        # print( *((sketch.evaluate([]), prob) for sketch, prob in choices), sep='\n')
        #print("time before hole reweight with dcmodel", time.time())
        choices = [( sketch, dcModel.infer_grammar((dc_input, tree_to_seq(sketch.evaluate([])))).sketchLogLikelihood(tsymbol, prog, sketch)[0] ) for sketch, prob in choices]  #TODO check this
        #print("time after hole reweight with dcmodel", time.time())
        # print("probs after grammar inf:")
        # print( *((sketch.evaluate([]), prob) for sketch, prob in choices), sep='\n')

    if len(list(choices)) == 0:
        #if there are none, then use the original program ,
        choices = [(prog, 0)]  # TODO
    progs, weights = zip(*choices)

    if sample_fn is None:
        sample_fn = lambda x: inv_temp*math.exp(inv_temp*x)

    if use_timeout:
        # sample timeout
        r = random.random()
        t = -math.log(r)/inv_temp

        cs = list(zip(progs, [-w for w in weights]))
        if t < list(cs)[0][1]: return prog, None, None

        below_cutoff_choices = [(p, w) for p,w in cs if t > w]

        _, max_w = max(below_cutoff_choices, key=lambda item: item[1])

        options = [(p, None, None) for p, w in below_cutoff_choices if w==max_w]
        x = random.choices(options, k=1)
        return x[0]
    else:
    #normalize weights, and then rezip
    
        weights = [sample_fn(w) for w in weights]
        #normalize_weights
        w_sum = sum(w for w in weights)
        weights = [w/w_sum for w in weights]
    

    if reward_fn is None:
        reward_fn = math.exp
    rewards = [reward_fn(w) for w in weights]

    prog_reward_probs = list(zip(progs, rewards, weights))

    if verbose:
        for p, r, prob in prog_reward_probs:
            print(p, prob)

    if k > 1:
        x = random.choices(prog_reward_probs, weights=weights, k=1)
        return x[0] #outputs prog, prob
    else:
        return prog_reward_probs[0] #outputs prog, prob


def tree_to_prog(tree):

    def init_arg_list(expr):
        #convert to symbol
        tp = expr.infer()
        if tp == tsymbol:
            symb = expr 
        elif tp == tconstant:
            symb = Application(Primitive.GLOBALS["symbol_constant"], expr)
        elif tp == tfunction:
            symb = Application(Primitive.GLOBALS["symbol_function"], expr)
        elif expr.isHole:
            symb = expr
        else:
            print("unsupported expression:")
            print("type:", tp)
            print("expr:", expr)
            assert False

        return Application(Primitive.GLOBALS["list_init_symbol"], symb) #TODO

    def add_to_list(args, expr):
        #convert to symbol
        tp = expr.infer()
        if tp == tsymbol:
            symb = expr 
        elif tp == tconstant:
            symb = Application(Primitive.GLOBALS["symbol_constant"], expr)
        elif tp == tfunction:
            symb = Application(Primitive.GLOBALS["symbol_function"], expr)
        elif expr.isHole:
            symb = expr
        else:
            print("unsupported expression:")
            print("type:", tp)
            print("expr:", expr)
            assert False

        return Application(Application(Primitive.GLOBALS["list_add_symbol"], symb), args)#TODO

    def get_fn_and_args(l):
        fn = recurse(l[0])
        if len(l) < 2: assert False
        args = init_arg_list(recurse(l[1]))
        for exp in l[2:]:
            args = add_to_list(args, recurse(exp))
        return fn, args

    def recurse(l):
        if type(l) == list:
        #then it is a function call or a lambda
            if l[0] == "lambda1":
                assert len(l) == 2
                in_list = l[1]
                fn, args = get_fn_and_args(in_list)
                return Application(Application(Primitive.GLOBALS['lambda1_call'], fn), args)

            elif l[0] == "lambda2":
                assert len(l) == 2
                in_list = l[1]
                fn, args = get_fn_and_args(in_list)
                return Application(Application(Primitive.GLOBALS['lambda2_call'], fn), args)

            else: # function call
                fn, args = get_fn_and_args(l)
                return Application(Application(Primitive.GLOBALS['fn_call'], fn), args)

        elif l in primitive_lookup:
            #these will be of type function or constant
            return Primitive.GLOBALS[primitive_lookup[l]]
        elif l == "lambda1":
            assert False
        elif l == "lambda2":
            assert False 
        elif l == '<HOLE>': #TODO
            return AlgolispHole()
        else:
            if not l==" ":
                print("l is not space")
                l.__repr__()
            if not l=="\t":
                print("l is not tab")
            assert False, f"uncaught item: {l}"
        #elif l == variable name:
        #    assert False

    return recurse(tree)


if __name__=='__main__':
    from grammar import Grammar
    from program import Program
    from algolispPrimitives import algolispProductions

    g = Grammar.fromProductions(algolispProductions(), logVariable=.9)

    #p=Program.parse("(lambda (fn_call filter (list_add_symbol (lambda1_call == (list_add_symbol 1 (list_init_symbol (fn_call mod ( list_add_symbol 2 (list_init_symbol arg1)) ))) ) (list_init_symbol $0)) )")
    p=Program.parse("(fn_call filter (list_add_symbol (lambda1_call eq (list_add_symbol (symbol_constant 1) (list_init_symbol (fn_call mod ( list_add_symbol (symbol_constant 2) (list_init_symbol (symbol_constant arg1))) ))) ) (list_init_symbol (symbol_constant a))))")

    print(p)

    #tree = p.evaluate(["a"])
    tree = p.evaluate([])
    print(tree)

    prog = tree_to_prog(tree)
    print(prog)




