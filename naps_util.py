#deepcoder_util.py
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
import sys
sys.path.append("/om/user/mnye/ec")
from grammar import Grammar, NoCandidates
from napsPrimitives import deepcoderPrimitives
from program import Application, Hole, Primitive, Index, Abstraction, ParseFailure
import math
from type import Context, arrow, tint, tlist, tbool, UnificationFailure

def uast_to_ec(uast:uast) -> Program:  # TODO
    # get list l
    def recurse(l):  # a list 
        if l in Primitive.GLOBALS:
            e = Primitive.GLOBALS[thing]  # TODO rename prims
        elif l in other_prims: # TODO fix this part
            e = other_prims[l]
        else:
            assert type(l) == list
            e = recurse(l[0])
            for i in l[1:]:
                e = Application(e, recurse(i))
        return e
    return ec_program

def tokenize_lisp_expr(sexp):
    #split sexp by spaces
    #split at brackets and parens
    return slist