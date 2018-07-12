#Sketch project


from builtins import super
import pickle
import string
import argparse

import torch
from torch import nn, optim

from pinn import RobustFill
import pregex as pre
from vhe import VHE, DataLoader, Factors, Result

from regex_prior import RegexPrior

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, choices=['qc','pc','px','vhe'])
args = parser.parse_args()

regex_prior = RegexPrior()
k_shot = 4
regex_vocab = list(string.printable[:-4]) + \
    [pre.OPEN, pre.CLOSE, pre.String, pre.Concat, pre.Alt, pre.KleeneStar, pre.Plus, pre.Maybe] + \
    regex_prior.character_classes


def lookup_str(string: str) -> ec.Program:
    pass

def pre_to_prog(regex: pre.pregex) -> ec.Program:

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





if __name__ == "__main__":
    print("Loading model", flush=True)
    try: model=torch.load("./sketch_model.p")
    except FileNotFoundError: model = RobustFill(input_vocabularies=[string.printable[:-4]], target_vocabulary=regex_vocab)


    ######## Pretraining ########
    batch_size = 500
    max_length = 15

    def getInstance():
        """
        Returns a single problem instance, as input/target strings
        """
        while True:
            r = regex_prior.sampleregex()
            c = r.flatten()
            x = r.sample()
            Dc = [r.sample() for i in range(k_shot)]
            c_input = [c]
            if all(len(x)<max_length for x in Dc + [c, x]): break
        return {'Dc':Dc, 'c':c, 'c_input':c_input, 'x':x}

    def getBatch():
        """
        Create a batch of problem instances, as tensors
        """
        instances = [getInstance() for i in range(batch_size)]
        Dc = [inst['Dc'] for inst in instances]
        c = [inst['c'] for inst in instances]
        c_input = [inst['c_input'] for inst in instances]
        x = [inst['x'] for inst in instances]
        return Dc, c, c_input, x

    if args.train:
        print("pretraining", flush=True)
        if not hasattr(model, 'iteration'):
            model.iteration = 0
            model.scores = []
        for i in range(model.iteration, 20000):
            Dc, c, c_input, x = getBatch()
            score = model.optimiser_step(Dc, c)

            model.scores.append(score)
            model.iteration += 1
            if i%10==0: print(args.train, "iteration", i, "score:", score, flush=True)
            if i%500==0: torch.save(f, './sketch_model.p')
    ######## End Pretraining ########


####### Pretrain with holes ########



#Kevin's RL version

#I think this is the right objective ... check https://arxiv.org/pdf/1506.05254.pdf
objective = score(sketch)*(logp(hole) - score(full_program).data) 

####### End pretrain with holes ########




###### full training ########
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for batch in actual_data: #or synthetic data, whatever:
    optimizer.zero_grad()
    samples, scores = model.sampleAndScore(batch, autograd=True) #TODO: change so you can get grad through here, and so that scores are seperate???
    objective = []
    for sample, score, examples in zip(samples, scores, batch):
        #DO RL
        objective.append = -score*find_ll_reward_with_enumeration(sample, examples, time=10) #TODO, ideally should be per hole


    objective = torch.sum(objective)
    #DO RL 
    #TODO???? oh god. Make reward in positive range??) 
    #Q: is it even

    objective.backward()
    optimizer.step()




#from holes, enumerate:

#option 1: use ec enumeration
#option 2: use something else

"""
RL questions:
- should I do param updates for each batch???
- can we even get gradients through the whole sample? no, but not too hard I think
- 
"""

#trees will be nice because you can enumerate within the tree by just sampling more --- then is it even worth it??


###### End full training ########



#informal testing:







