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
import math
#from util import enumerate_reg, Hole

from grammar import Grammar, NoCandidates
from napsPrimitives import napsPrimitives
from program import Application, Hole, Primitive, Index, Abstraction, ParseFailure, Program
from type import Context, arrow, tint, tlist, tbool, UnificationFailure

primitive_lookup = {prim.name:prim.name for prim in napsPrimitives()}
primitive_lookup['val'] = "constant"
primitive_lookup['?:'] = "ternary"


type_lookup = {
    '*': "array",
    '%': "set",
    '<|>': "map",
    "record_name#":"record_name"
    }

def convert_to_tp(x, target_tp):
    original_tp = x.infer()
    request = arrow(original_tp, target_tp)
    converter = [x for x in Primitive.GLOBALS.values() if x.tp==request][0]  # TODO - hack
    return Application(converter, x)

def uast_to_ec(uast) -> Program:  # TODO
    # get list l

    def recurse_list(l, target_tp):
        #base case
        x = recurse(l[0])
        x = convert_to_tp(x, target_tp)
        e = convert_to_tp(x, tlist(target_tp))
        for exp in l[1:]:
            x = recurse(exp)
            if x.infer() != target_tp:
                x = convert_to_tp(x, target_tp)  # maybe always convert?
            request = arrow(target_tp, tlist(target_tp), tlist(target_tp)) 
            list_converter = [x for x in Primitive.GLOBALS.values() if x.tp==request][0]  # TODO
            e = Application(Application(list_converter, x), e)
        return e

    def recurse(l):  # a list 
        if type(l) == list:
            e = recurse(l[0])
            tp_args = e.infer().functionArguments()
            for tp_arg, exp in zip(tp_args, l[1:]):
                if tp_arg.name=='list':  
                    # for now, assume the correct num of brackets
                    x = recurse_list(exp, tp_arg.arguments[0])
                else:
                    x = recurse(exp)
                    if tp_arg != x.infer():
                        x = convert_to_tp(x, tp_arg)
                e = Application(e, x)
            return e
        elif l in primitive_lookup:
            e = Primitive.GLOBALS[primitive_lookup[l]] 
        elif l in type_lookup: # TODO fix this part
            raise unimplemented()
        # elif 
        #     e = other_prims[l]
        # elif l in names_stuff:
        #     raise unimplemented()
        elif type(l) == int:
            l = str(l)
            e = recurse(l)
        else:
            print("l:", l)
            assert False
        return e

    return recurse(uast) # ec_program

def tokenize_lisp_expr(sexp):
    #split sexp by spaces
    #split at brackets and parens
    return slist

if __name__=='__main__':
    #["assign", "bool", ["var", "bool", "var5"], ["val", "bool", False]]
    u = ["invoke", "bool", "&&", [["invoke", "bool", "!", [["invoke", "bool", "!=", [["val", "int", -1], ["invoke", "int", "string_find", [["var", "char*", "var0"], ["val", "char*", "1"]]]]]]], ["invoke", "bool", "!=", [["val", "int", -1], ["invoke", "int", "string_find", [["var", "char*", "var1"], ["val", "char*", "1"]]]]]]]
    p = uast_to_ec(u)
    print(p)
    e = p.evaluate([])
    print(e)
    print(u==e)

