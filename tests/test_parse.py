#Sketch project


#from builtins import super
#import pickle
#import string
#import argparse
#import random

#import torch
#from torch import nn, optim

#from pinn import RobustFill
#from pinn import SyntaxCheckingRobustFill #TODO
#import random
#import math

#from collections import OrderedDict
#from util import enumerate_reg, Hole

import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./ec'))

from grammar import Grammar
from deepcoderPrimitives import deepcoderProductions, flatten_program

#from program import Application, Hole

#import math
from type import Context, arrow, tint, tlist, tbool, UnificationFailure
from program import prettyProgram

from train.main_supervised_deepcoder import parseprogram, make_holey_deepcoder

#g = Grammar.uniform(deepcoderPrimitives())
g = Grammar.fromProductions(deepcoderProductions(), logVariable=.9) #TODO - find correct grammar weights
request = arrow(tlist(tint), tint, tint)

p = g.sample(request)

sketch = make_holey_deepcoder(p, 10, g, request)

print("request:", request)
print("program:")
print(prettyProgram(p))
print("flattened_program:")
flat = flatten_program(p)
print(flat)

prog = parseprogram(flat, request)
print("recovered program:")
print(prettyProgram(prog))
print("-----")
print("sketch:")
print(sketch)
print("flattend sketch:")
flatsketch = flatten_program(sketch)
print(flatsketch)
print("recovered sketch")
recovered_sketch = parseprogram(flatsketch, request)
print(recovered_sketch)


for i in range(1000):
	p = g.sample(request)

	sketch = make_holey_deepcoder(p, 10, g, request)


	flat = flatten_program(p)


	prog = parseprogram(flat, request)

	flatsketch = flatten_program(sketch)

	recovered_sketch = parseprogram(flatsketch, request)
	if not flatsketch == ['<HOLE>']:
		assert recovered_sketch == sketch
