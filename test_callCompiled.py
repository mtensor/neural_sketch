#test callCompiled
import sys
sys.path.append("/om/user/mnye/ec")
from utilities import callCompiled, eprint

from fun import f

x = 6
ans = callCompiled(f, x)

eprint(ans)