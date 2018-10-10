#test callCompiled
import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./ec'))

from utilities import callCompiled, eprint

from fun import f

x = 6
ans = callCompiled(f, x)

eprint(ans)