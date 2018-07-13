#utils.py
"""
utils for neural_sketch, mostly conversions between pregex and EC. unused for now, may be important later 

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
        raise unimplemented
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