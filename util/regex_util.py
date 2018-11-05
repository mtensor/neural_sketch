#utils.py
"""
utils for neural_sketch, mostly conversions between pregex and EC. unused for now, may be important later 

"""
import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./ec'))

from grammar import Grammar
from regexPrimitives import sketchPrimitives, concatPrimitives, disallowed
import math
from type import tpregex, Context
from program import Hole, Primitive, Application

import pregex as pre
import dill
import random
from vhe import RegexPrior
from string import printable

regex_prior = RegexPrior()

def concatProductions():
    prim_list = concatPrimitives()
    specials = ["r_kleene", "r_plus", "r_maybe", "r_alt"]
    n_base_prim = len(prim_list) - len(specials)

    productions = [
        (math.log(0.6 / float(n_base_prim)),
         prim) if prim.name not in specials else (
            math.log(0.10),
            prim) for prim in prim_list]
    return productions


class PregHole(pre.Pregex):
    def __new__(cls): return super(PregHole, cls).__new__(cls, None)
    def __repr__(self): return "(HOLE)"
    def flatten(self, char_map={}, escape_strings=False):
        return [type(self)]
    def walk(self, depth=0):
        """
        walks through the nodes
        """
        yield self, depth

#EC-land hole 
class RegexHole(Hole):
    def show(self, isFunction): return "<RegexHOLE>"

    def evaluate(self, e):
        return PregHole()

    @staticmethod
    def _parse(s,n):
        while n < len(s) and s[n].isspace(): n += 1
        n = Program.parseConstant(s,n,
                                  '<RegexHOLE>')
        return RegexHole(), n




#basegrammar = Grammar.fromProductions(concatProductions(), logVariable=math.log(0.2)) # TODO -- fix this!!!

prim_list = sketchPrimitives()
specials = ["r_kleene", "r_plus", "r_maybe", "r_alt", "r_concat"]
n_base_prim = len(prim_list) - len(specials)

productions = [
    (math.log(0.5 / float(n_base_prim)),
     prim) if prim.name not in specials else (
        math.log(0.10),
        prim) for prim in prim_list]

basegrammar = Grammar.fromProductions(productions)


print("WARNGING: base grammar is very naive")


def tokenize_for_robustfill(IOs):
    """
    tokenizes a batch of IOs ... I think none is necessary ...
    """
    return IOs

def sample_program(g, request):
    return pre_to_prog(regex_prior.sampleregex())

def generate_IO_examples(p, num_examples=5, continuation=False):

    if continuation:
        preg = p.evaluate([])(pre.String(""))
    else:
        preg = p.evaluate([])
    return [preg.sample() for _ in range(num_examples)]
    #TODO


def flatten_program(p, continuation=False):
    # version which flattens programs naively
    if continuation:
        return p.evaluate([])(pre.String("")).flatten()
    else: 
        return p.evaluate([]).flatten()
        
def make_holey_regex(prog, k, g, request, inv_temp=1.0, reward_fn=None, sample_fn=None, verbose=False, use_timeout=False):
    """
    inv_temp==1 => use true mdls
    inv_temp==0 => sample uniformly
    0 < inv_temp < 1 ==> something in between
    """ 
    choices = g.enumerateHoles(request, prog, k=k, return_obj=RegexHole)

    if len(list(choices)) == 0:
        #if there are none, then use the original program 
        choices = [(prog, 0)]
    #print("prog:", prog, "choices", list(choices))
    progs, weights = zip(*choices)

    # if verbose:
    #     for c in choices: print(c)

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



disallowed_dict = {char: "string_"+name for char, name in disallowed}
allowed_dict = {i:"string_" + i for i in printable[:-4] if i not in disallowed_dict}
lookup_dict = {**disallowed_dict, **allowed_dict}


def lookup_str(string: str):
    name = lookup_dict[string]
    return Primitive.GLOBALS[name]



def pre_to_prog(regex):
    #print("WARNING: this function is completely untested.")
    #also this context, environment, request stuff .... 

    #print("need to deal with concat and alts with more than two inputs")
    if regex.type == 'Concat':
        if len(regex.values) == 2:
            return Application(Application(Primitive.GLOBALS['r_concat'], pre_to_prog(regex.values[0])), pre_to_prog(regex.values[1]) ) 
        else:
            e = Application(Application(Primitive.GLOBALS['r_concat'], pre_to_prog(regex.values[0])), pre_to_prog(regex.values[1]))
            return Application(Application(Primitive.GLOBALS['r_concat'], e), pre_to_prog(pre.String(regex.values[2:])))
    elif regex.type == 'KleeneStar':
        return Application(Primitive.GLOBALS['r_kleene'], pre_to_prog(regex.val))
    elif regex.type == 'Alt':
        if len(regex.values) == 2:
            return Application(Application(Primitive.GLOBALS['r_alt'], pre_to_prog(regex.values[0]), pre_to_prog(regex.values[1]) ) )
        else:
            e = Application(Application(Primitive.GLOBALS['r_alt'], pre_to_prog(regex.values[0])), pre_to_prog(regex.values[1]))
            return Application(Application(Primitive.GLOBALS['r_alt'], e), pre_to_prog(pre.String(regex.values[2:])))
            
    elif regex.type == 'Plus':
        return Application(Primitive.GLOBALS['r_plus'], pre_to_prog(regex.val))
    elif regex.type == 'Maybe':
        return Application(Primitive.GLOBALS['r_maybe'], pre_to_prog(regex.val))
    elif regex.type == 'String':
        #print("WARNING: doing stupidest possible thing for Strings")
        if len(regex.arg) == 1:
            return lookup_str(regex.arg[0]) 
        else:
            e = lookup_str(regex.arg[0]) #Application(Application(Primitive.GLOBALS['r_concat'], pre_to_prog(regex.values[0])), pre_to_prog(regex.values[1]))
            return Application(Application(Primitive.GLOBALS['r_concat'], e), pre_to_prog(pre.String(regex.arg[1:])))
            
            #for v in regex.arg[1:]:
            #    e = Application(Application(e, Primitive.GLOBALS['r_concat']), pre_to_prog(pre.String(v)))
            #return e

            #return Application(Application(Primitive.GLOBALS['r_concat'], lookup_str(regex.arg[0])), pre_to_prog(pre.String(regex.arg[1:]) )) # TODO
    elif regex.type == 'PregHole':
        return RegexHole() #TODO
    elif regex.type == 'CharacterClass':
        if regex.name ==  '.': return Primitive.GLOBALS['r_dot']
        elif regex.name ==  '\\d': return Primitive.GLOBALS['r_d']
        elif regex.name ==  '\\s': return Primitive.GLOBALS['r_s']
        elif regex.name ==  '\\w': return Primitive.GLOBALS['r_w']
        elif regex.name ==  '\\l': return Primitive.GLOBALS['r_l']
        elif regex.name ==  '\\u': return Primitive.GLOBALS['r_u']
        else: assert False
    else: assert False

if __name__ == '__main__':
    preg = regex_prior.sampleregex()

    print(preg)
    print(preg.__repr__())

    prog = pre_to_prog(preg) 
    print(prog)

    preg2 = prog.evaluate([])
    print(preg2)


    # print("testing enumeration")
    # print("creating the enum_dict")
    # from collections import OrderedDict
    # d = {r: s for r, s in enumerate_reg(13)} #TODO #13 gives 24888

    # #sort dict
    # enum_dict = OrderedDict(sorted(d.items(), key=lambda s: -s[1]))

    # print(enum_dict)


# def make_holey(r: pre.Pregex, p=0.05) -> (pre.Pregex, torch.Tensor):
#     """
#     makes a regex holey
#     """
#     scores = 0
#     def make_holey_inner(r: pre.Pregex) -> pre.Pregex:
#         if random.random() < p: 
#             nonlocal scores
#             scores += regex_prior.scoreregex(r)
#             return Hole()
#         else: 
#             return r.map(make_holey_inner)

#     holey = make_holey_inner(r)
#     return holey, torch.Tensor([scores])

# def sketch_logprior(preg: pre.Pregex, p=0.05) -> torch.Tensor:
#     logprior=0
#     for r, d in preg.walk():
#         if type(r) is pre.String or type(r) is pre.CharacterClass or type(r) is Hole: #TODO, is a leaf
#             if type(r) is Hole: #TODO
#                 logprior += math.log(p) + d*math.log(1-p)
#             else:
#                 logprior += (d+1)*math.log(1-p)

#     return torch.tensor([logprior])

# def make_holey_supervised(reg: pre.Pregex, enum_dict: dict, k=1) -> (pre.Pregex, float): #second return val should be torch.Tensor
#     """
#     makes a regex holey in a supervised way 
#     right now, grabs the regex with the best score
#     grab top k
#     we require that the enum_dict be ordered
#     """
#     enum_list = list(enum_dict)
#     done = False
#     scores = []
#     def substitute_once(r: pre.Pregex) -> pre.Pregex:
#         #REMINDER, there is the sub argument which is required in here
#         nonlocal done
#         nonlocal scores
#         if done:
#             return r
#         else: 
#             if r == sub:
#                 done = True
#                 scores.append(enum_dict[sub])
#                 return Hole()
#             else: 
#                 return r.map(substitute_once) #TODO

#     solutions = []
#     for sub in enum_list: 
#         while True:
#             if len(solutions) == k: 
#                 assert k == len(scores)
#                 return tuple(zip(solutions, scores))
#             solution = substitute_once(reg)
#             if done:
#                 done = False
#                 solutions.append(solution)
#             else: break

#     if len(solutions) == 0:
#         return tuple(((reg, 0.0),))

#     assert len(solutions) == len(scores)
#     return tuple(zip(solutions, scores))

    # holey = make_holey_sup_inner(r)
    # return holey, torch.Tensor([scores])


# prim_list = sketchPrimitives()
# specials = ["r_kleene", "r_plus", "r_maybe", "r_alt", "r_concat"]
# n_base_prim = len(prim_list) - len(specials)

# productions = [
#     (math.log(0.5 / float(n_base_prim)),
#      prim) if prim.name not in specials else (
#         math.log(0.10),
#         prim) for prim in prim_list]

# basegrammar = Grammar.fromProductions(productions)

# def enumerate_reg(number):
#     depth = number
#     yield from ((prog.evaluate([]), l) for l, _, prog in basegrammar.enumeration(Context.EMPTY, [], tpregex, depth))

# def fill_hole(sketch:pre.pregex, subtree:pre.pregex) -> pre.pregex:
#     """
#     a function which fills one hole WITH THE SAME SUBTREE. requires only one hole in the whole thing
#     """
#     def fill_hole_inner(sketch:pre.pregex) -> pre.pregex:
#         if type(sketch) is Hole: return subtree
#         else: return sketch.map(fill_hole_inner)
#     return fill_hole_inner(sketch)

# ####data loading#####
# def date_data(maxTasks=None, nExamples=5):
#     taskfile = "./dates.p"
#     with open(taskfile, 'rb') as handle:
#         data = dill.load(handle)
#     tasklist =  [column['data'][:nExamples] for column in data if len(column['data']) >= nExamples]
#     if maxTasks is not None:
#         random.seed(42) #42 #80
#         random.shuffle(tasklist)
#         del tasklist[maxTasks:]
#     return tasklist

# def all_data(maxTasks=None, nExamples=5):
#     taskfile = "./regex_data_csv_900.p"
#     with open(taskfile, 'rb') as handle:
#         data = dill.load(handle)
#     tasklist = [column[:nExamples] for column in data[0] if len(column) >=nExamples] #a list of indices
#     if maxTasks is not None:
#         random.seed(42) #42 #80
#         random.shuffle(tasklist)
#         del tasklist[maxTasks:]
#     return tasklist

"""


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


