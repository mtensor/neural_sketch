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

from train.main_supervised_deepcoder import parseprogram, make_holey_deepcoder, sampleIO, getInstance, grammar
import time

max_length=30

inst = getInstance(5, verbose=True)
print("program:")
print(inst['p'])
print("IO:")
print(inst['IO'])

t = time.time()
p = make_holey_deepcoder(inst['p'], 5, grammar, inst['tp'])
print(time.time() - t)

t = time.time()
sketch = flatten_program(p)
print(time.time() - t)