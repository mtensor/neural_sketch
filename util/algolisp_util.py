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
import math
import time

from collections import OrderedDict
#from util import enumerate_reg, Hole
from grammar import Grammar, NoCandidates
from program import Application, Hole, Primitive, Index, Abstraction, ParseFailure, HoleFinder, HolePuncher

import math
from type import Context, arrow, tint, tlist, tbool, UnificationFailure
from algolispPrimitives import tsymbol, tconstant, tfunction, primitive_lookup
from deepcoderPrimitives import flatten_program

from program_synthesis.algolisp.dataset import data
from program_synthesis.algolisp.dataset import executor

from collections import OrderedDict
#primitive_lookup = {prim.name:prim.name for prim in napsPrimitives()}

#get dict of tree: mdl, location
#given dict, find best sketches

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

def no_path_conflict(sketch, subtree) -> bool:
    l2 = len(subtree.path)
    for tr in sketch:
        l1 = len(tr.path)
        if l1 > l2:
            conflict = (tr.path[:l2] == subtree.path)
        else:
            conflict = (tr.path == subtree.path[:l1])
        if conflict:
            return False
    return True

def place_hole(g, p, subtree, return_obj=AlgolispHole):
    return HolePuncher(g, subtree.path, return_obj()).execute(p)

def concretize_sketches_w_mdl(g, sketches, p, tp, return_obj=AlgolispHole):
    #do some stuff
    choices = [(p, 0.0)]
    for sketch in sketches:
        choice = p
        for subtree in sketch:
            choice = place_hole(g, choice, subtree, return_obj=return_obj)
        choices.append((choice, mdl_calc(sketch)))
    return choices

def mdl_calc(sketch):
    return sum(subtree.mdl for subtree in sketch)

def findsubtrees(g, p, request, k=3, return_obj=Hole):
    """Enumerate programs with a single hole within mdl distance"""
    #TODO: make it possible to enumerate sketches with multiple holes
    def mutations(tp, loss, is_left_application=False):
        """
        to allow applications lhs to become a hole,  
        remove the condition below and ignore all the is_left_application kwds 
        """
        if not is_left_application: 
            yield return_obj(), 0
    top_k = []
    for subtree in HoleFinder(g, mutations).execute(p, request):
        if len(top_k) > 0:
            i, v = min(enumerate(top_k), key=lambda x:x[1].mdl)
            if subtree.mdl > v.mdl:
                if len(top_k) >= k:
                    top_k[i] = subtree
                else:
                    top_k.append(subtree)
            elif len(top_k) < k:
                top_k.append(subtree)
        else:
            top_k.append(subtree)
    return sorted(top_k, key=lambda x:-x.mdl)

def top_k_sketches(g, k, nHoles, p, tp, return_obj=AlgolispHole):

    """
    [X] make HoleFinder output namedtuple 
    [X] run HoleFinder here
    [X] rewrite path so that it can be compared w other paths
    [X] sort output of HoleFinder
    [ ] make sure you also output the sketch with no holes
    [ ] test everything
    [ ] put in correct place in code

    """
    mem = findsubtrees(g, p, tp, k=k, return_obj=return_obj) #tho it doesn't matter
    #TURN INTO SUBTREE OBJECTS
    #print("k",k)
    sketches = [ (subtree,) for subtree in mem]
    min_mdl = mdl_calc(sketches[-1])
    for _ in range(1, nHoles):
        additions = []
        for sketch in sketches:
            for subtree in mem:
                if mdl_calc(sketch + (subtree,)) < min_mdl: break
                if no_path_conflict(sketch, subtree): 
                    additions.append(sketch + (subtree,) )
        sketches = sorted(sketches+additions, key=lambda x: -mdl_calc(x))[:k] #TODO
        min_mdl = mdl_calc(sketches[-1])
    #print("done loop")
    choices = concretize_sketches_w_mdl(g, sketches, p, tp, return_obj=return_obj)

    return choices


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

def make_holey_algolisp(prog,
                        k,
                        request,
                        basegrammar,
                        dcModel=None,
                        improved_dc_model=False,
                        return_obj=AlgolispHole,
                        dc_input=None,
                        inv_temp=1.0,
                        reward_fn=None,
                        sample_fn=None,
                        verbose=False,
                        use_timeout=False,
                        nHoles=4,
                        use_fixed_seed=False,
                        rng=None,
                        domain='algolisp'):
    """
    inv_temp==1 => use true mdls
    inv_temp==0 => sample uniformly
    0 < inv_temp < 1 ==> something in between
    """ 
    if dcModel is None:
        #print("dcModel NONE")
        g = basegrammar
        #choices = g.enumerateMultipleHoles(request, prog, k=k, return_obj=return_obj, nHoles=nHoles)
        choices = top_k_sketches(g, k, nHoles, prog, request, return_obj=return_obj)
    elif dcModel and not improved_dc_model:
        g = dcModel.infer_grammar(dc_input) #(spec, sketch)
        #choices = g.enumerateMultipleHoles(request, prog, k=k, return_obj=return_obj, nHoles=nHoles)
        choices = top_k_sketches(g, k, nHoles, prog, request, return_obj=return_obj)
    else: 
        assert improved_dc_model
        g = basegrammar
        
        choices = top_k_sketches(g, k, nHoles, prog, request, return_obj=return_obj)
        
        # REWRITE:
        if domain=='algolisp':
            choices = [( sketch, dcModel.infer_grammar((dc_input, tree_to_seq(sketch.evaluate([])))).sketchLogLikelihood(tsymbol, prog, sketch)[0] ) for sketch, prob in choices] 
        elif domain=='list' or domain=='text':
            # print("debug prints:")
            # print([sketch for sketch, _ in choices])
            # print(dc_input)
            # print(prog)
            # print(request)
            #t = time.time()
            choices = [( sketch, dcModel.infer_grammar((dc_input, tuple(flatten_program(sketch)) )).sketchLogLikelihood(request, prog, sketch)[0] ) for sketch, prob in choices]  #TODO check this
            #print("time for top_k_sketches:", time.time() - t)
        else:
            assert False

    if len(list(choices)) == 0:
        #if there are none, then use the original program ,
        choices = [(prog, 0)]  # TODO
    progs, weights = zip(*choices)

    if sample_fn is None:
        sample_fn = lambda x: inv_temp*math.exp(inv_temp*x)

    if use_timeout:
        # sample timeout
        r = random.random() if not use_fixed_seed else rng.random()
        t = -math.log(r)/inv_temp

        cs = list(zip(progs, [-w for w in weights]))
        if t < list(cs)[0][1]: return prog, None, None

        below_cutoff_choices = [(p, w) for p,w in cs if t > w]

        _, max_w = max(below_cutoff_choices, key=lambda item: item[1])

        options = [(p, None, None) for p, w in below_cutoff_choices if w==max_w]
        x = random.choices(options, k=1) if not use_fixed_seed else rng.choices(options, k=1)
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
        x = random.choices(prog_reward_probs, weights=weights, k=1) if not use_fixed_seed else rng.choices(prog_reward_probs, weights=weights, k=1)
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




