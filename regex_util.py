#utils.py
"""
utils for neural_sketch, mostly conversions between pregex and EC. unused for now, may be important later 

"""
import sys
sys.path.append("/om/user/mnye/ec")

from grammar import Grammar
from regexPrimitives import sketchPrimitives
import math
from type import tpregex, Context

import pregex as pre
from sketch_project import Hole
import dill
import random

class Hole(pre.Pregex):
    def __new__(cls): return super(Hole, cls).__new__(cls, None)
    def __repr__(self): return "(HOLE)"
    def flatten(self, char_map={}, escape_strings=False):
        return [type(self)]
    def walk(self, depth=0):
        """
        walks through the nodes
        """
        yield self, depth


prim_list = sketchPrimitives()
specials = ["r_kleene", "r_plus", "r_maybe", "r_alt", "r_concat"]
n_base_prim = len(prim_list) - len(specials)

productions = [
    (math.log(0.5 / float(n_base_prim)),
     prim) if prim.name not in specials else (
        math.log(0.10),
        prim) for prim in prim_list]


baseGrammar = Grammar.fromProductions(productions)

def enumerate_reg(number):
    depth = number
    yield from ((prog.evaluate([]), l) for l, _, prog in baseGrammar.enumeration(Context.EMPTY, [], tpregex, depth))


def fill_hole(sketch:pre.pregex, subtree:pre.pregex) -> pre.pregex:
    """
    a function which fills one hole WITH THE SAME SUBTREE. requires only one hole in the whole thing
    """

    def fill_hole_inner(sketch:pre.pregex) -> pre.pregex:
        if type(sketch) is Hole:
            return subtree
        else:
            return sketch.map(fill_hole_inner)

    return fill_hole_inner(sketch)


####data loading#####
def date_data(maxTasks=None, nExamples=5):
    taskfile = "./dates.p"

    with open(taskfile, 'rb') as handle:
        data = dill.load(handle)

    tasklist =  [column['data'][:nExamples] for column in data if len(column['data']) >= nExamples]

    if maxTasks is not None:
        random.seed(42) #42 #80
        random.shuffle(tasklist)
        del tasklist[maxTasks:]

    return tasklist

def all_data(maxTasks=None, nExamples=5):
    taskfile = "./regex_data_csv_900.p"

    with open(taskfile, 'rb') as handle:
        data = dill.load(handle)

    tasklist = [column[:nExamples] for column in data[0] if len(column) >=nExamples] #a list of indices

    if maxTasks is not None:
        random.seed(42) #42 #80
        random.shuffle(tasklist)
        del tasklist[maxTasks:]

    return tasklist



"""
def lookup_str(string: str) -> ec.Program:
    pass

def pre_to_prog(regex: pre.pregex) -> ec.Program:
    print("WARNING: this function is completely untested.")
    #also this context, environment, request stuff .... 

    if regex.type == 'Concat':
        return Application(Primitive.GLOBALS['r_concat'], Application(pre_to_prog(regex.values[0]),pre_to_prog(regex.values[1]) ) )
    elif regex.type == 'KleeneStar':
        return Application(Primitive.GLOBALS['r_kleene'], pre_to_prog(regex.val))
    elif regex.type == 'Alt':
        return Application(Primitive.GLOBALS['r_alt'], Application(pre_to_prog(regex.values[0]), pre_to_prog(regex.values[1]) ) )
    elif regex.type == 'Plus':
        return Application(Primitive.GLOBALS['r_plus'], pre_to_prog(regex.val))
    elif regex.type == 'Maybe':
        return Application(Primtive.GLOBALS['r_maybe'], pre_to_prog(regex.val))
    elif regex.type == 'String':
        print("WARNING: doing stupidest possible thing for Strings")
        return Application(Primitive.GLOBALS['r_concat'], Application( lookup_str(regex.arg[0]), pre_to_prog(pre.String(regex.arg[0:]) )))
    elif regex.type == 'Hole':
        return Hole #TODO
    elif regex.type == 'CharacterClass':
        if regex.name ==  '.': return Primtive.GLOBALS['r_dot']
        elif regex.name ==  '\\d': return Primtive.GLOBALS['r_d']
        elif regex.name ==  '\\s': return Primtive.GLOBALS['r_s']
        elif regex.name ==  '\\w': return Primtive.GLOBALS['r_w']
        elif regex.name ==  '\\l': return Primtive.GLOBALS['r_l']
        elif regex.name ==  '\\u': return Primtive.GLOBALS['r_u']
        else: assert False
    else: assert False


def convert_ec_program_to_pregex(program: ec.program) -> pre.pregex:
    #probably just a conversion:
    return program.evaluate([]) #with catches, i think 

def find_ll_reward_with_enumeration(sample, examples, time=10):

#something like this: TODO:
    maxll = float('-inf')
    #make sample into a context maybe??
    contex = something(sample) #TODO
    environment = something_else #TODO
    request = tpregex #I think, TODO
    for prior, _, p in g.enumeration(Context.EMPTY, [], request,
                                             maximumDepth=99,
                                             upperBound=budget,
                                             lowerBound=previousBudget): #TODO:fill it out 
        ll = likelihood(p,examples) #TODO probably just convert to a pregex and then sum the match
        if ll > maxll:
            maxll = ll
        if timeout is not None and time() - starting > timeout:
            break
    return maxll

"""

if __name__ == '__main__':
    print("testing enumeration")
    print("creating the enum_dict")
    from collections import OrderedDict
    d = {r: s for r, s in enumerate_reg(13)} #TODO #13 gives 24888

    #sort dict
    enum_dict = OrderedDict(sorted(d.items(), key=lambda s: -s[1]))

    print(enum_dict)

