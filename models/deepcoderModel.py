#deepcodermodel
import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./ec'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimization
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import random


#from recognition import RecurrentFeatureExtractor, RecognitionModel
from grammar import Grammar  #?

from RobustFillPrimitives import RobustFillProductions
from string import printable

from recognition import GrammarNetwork, ContextualGrammarNetwork, RecognitionModel
from grammar import ContextualGrammar
from util.robustfill_util import robustfill_vocab

from util.algolisp_util import tokenize_for_dc

#from main_supervised_deepcoder import deepcoder_io_vocab #TODO
def _relu(x): return x.clamp(min=0)

def variable(x, volatile=False, cuda=False):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x, (np.ndarray, np.generic)):
        x = torch.from_numpy(x)
    if cuda:
        x = x.cuda()
    return Variable(x, volatile=volatile)


class RecurrentFeatureExtractor(nn.Module):
    def __init__(self, _=None,
                 cuda=False,
                 # what are the symbols that can occur in the inputs and
                 # outputs
                 lexicon=None,
                 # how many hidden units
                 H=32,
                 # Should the recurrent units be bidirectional?
                 bidirectional=False):
        super(RecurrentFeatureExtractor, self).__init__()

        #assert tasks is not None, "You must provide a list of all of the tasks, both those that have been hit and those that have not been hit. Input examples are sampled from these tasks."

        # maps from a requesting type to all of the inputs that we ever saw
        # that request
        # self.requestToInputs = {
        #     tp: [
        #         list(
        #             map(
        #                 fst,
        #                 t.examples)) for t in tasks if t.request == tp] for tp in {
        #         t.request for t in tasks}}

        assert lexicon
        self.specialSymbols = [
            "STARTING",  # start of entire sequence
            "ENDING",  # ending of entire sequence
            "STARTOFOUTPUT",  # begins the start of the output
            "ENDOFINPUT"  # delimits the ending of an input - we might have multiple inputs
        ]
        lexicon += self.specialSymbols
        encoder = nn.Embedding(len(lexicon), H)
        if cuda:
            encoder = encoder.cuda()
        self.encoder = encoder

        self.H = H
        self.bidirectional = bidirectional

        layers = 1

        model = nn.GRU(H, H, layers, bidirectional=bidirectional)
        if cuda:
            model = model.cuda()
        self.model = model

        self.use_cuda = cuda
        self.lexicon = lexicon
        self.symbolToIndex = {
            symbol: index for index,
            symbol in enumerate(lexicon)}
        self.startingIndex = self.symbolToIndex["STARTING"]
        self.endingIndex = self.symbolToIndex["ENDING"]
        self.startOfOutputIndex = self.symbolToIndex["STARTOFOUTPUT"]
        self.endOfInputIndex = self.symbolToIndex["ENDOFINPUT"]

        # Maximum number of inputs/outputs we will run the recognition
        # model on per task
        # This is an optimization hack
        self.MAXINPUTS = 100

    @property
    def outputDimensionality(self): return self.H

    # modify examples before forward (to turn them into iterables of lexicon)
    # you should override this if needed
    def tokenize(self, x): return x

    def symbolEmbeddings(self):
        return {s: self.encoder(variable([self.symbolToIndex[s]])).squeeze(
            0).data.numpy() for s in self.lexicon if not (s in self.specialSymbols)}

    def packExamples(self, examples):
        """IMPORTANT! xs must be sorted in decreasing order of size because pytorch is stupid"""
        es = []
        sizes = []
        for xs, y in examples:
            e = [self.startingIndex]
            for x in xs:
                for s in x:
                    e.append(self.symbolToIndex[s])
                e.append(self.endOfInputIndex)
            e.append(self.startOfOutputIndex)
            for s in y:
                e.append(self.symbolToIndex[s])
            e.append(self.endingIndex)
            if es != []:
                assert len(e) <= len(es[-1]), \
                    "Examples must be sorted in decreasing order of their tokenized size. This should be transparently handled in recognition.py, so if this assertion fails it isn't your fault as a user of EC but instead is a bug inside of EC."
            es.append(e)
            sizes.append(len(e))

        m = max(sizes)
        # padding
        for j, e in enumerate(es):
            es[j] += [self.endingIndex] * (m - len(e))

        x = variable(es, cuda=next(self.encoder.parameters()).is_cuda) #checks if encoder is cuda'd or not
        x = self.encoder(x)
        # x: (batch size, maximum length, E)
        x = x.permute(1, 0, 2)
        # x: TxBxE
        x = pack_padded_sequence(x, sizes)
        return x, sizes

    def examplesEncoding(self, examples):
        examples = sorted(examples, key=lambda xs_y: sum(
            len(z) + 1 for z in xs_y[0]) + len(xs_y[1]), reverse=True)
        x, sizes = self.packExamples(examples)
        self.model.flatten_parameters()
        outputs, hidden = self.model(x)
        # outputs, sizes = pad_packed_sequence(outputs)
        # I don't know whether to return the final output or the final hidden
        # activations...
        return hidden[0, :, :] + hidden[1, :, :]

    def forward(self, examples):
        tokenized = self.tokenize(examples)
        if not tokenized:
            return None

        if hasattr(self, 'MAXINPUTS') and len(tokenized) > self.MAXINPUTS:
            tokenized = list(tokenized)
            random.shuffle(tokenized)
            tokenized = tokenized[:self.MAXINPUTS]
        e = self.examplesEncoding(tokenized)
        # max pool
        # e,_ = e.max(dim = 0)

        # take the average activations across all of the examples
        # I think this might be better because we might be testing on data
        # which has far more o far fewer examples then training
        e = e.mean(dim=0)
        return e

    def featuresOfTask(self, t):
        f = self(t.examples) 
        if f is None:
            eprint(t)
        return f

    def featuresOfExamples(self, examples): #max added this to bypass any EC stuff
        return self(examples)

    # def taskOfProgram(self, p, tp):
    #     candidateInputs = list(self.requestToInputs[tp])
    #     # Loop over the inputs in a random order and pick the first one that
    #     # doesn't generate an exception
    #     random.shuffle(candidateInputs)
    #     for xss in candidateInputs:
    #         ys = []

    #         for xs in xss:
    #             try:
    #                 y = runWithTimeout(lambda: p.runWithArguments(xs),0.01)
    #             except:
    #                 break

    #             ys.append(y)
    #         if len(ys) == len(xss):
    #             return Task("Helmholtz", tp, list(zip(xss,ys)))

    #     return None

#For DEEPCODER TASKS
class LearnedFeatureExtractor(RecurrentFeatureExtractor):

    def tokenize(self, examples):
        def sanitize(l): return [z if z in self.lexicon else "?"
                                 for z_ in l
                                 for z in (z_ if isinstance(z_, list) else [z_])]
        tokenized = []
        for xs, y in examples:
            if isinstance(y, list):
                y = ["LIST_START"] + y + ["LIST_END"]
            else:
                y = [y]
            y = sanitize(y)
            if len(y) > self.maximumLength:
                return None

            serializedInputs = []
            for xi, x in enumerate(xs):
                if isinstance(x, list):
                    x = ["LIST_START"] + x + ["LIST_END"]
                else:
                    x = [x]
                x = sanitize(x)
                if len(x) > self.maximumLength:
                    return None
                serializedInputs.append(x)
            tokenized.append((tuple(serializedInputs), y))
        return tokenized

    def __init__(self, lexicon, hidden=128, use_cuda=True):
        self.H = hidden
        self.USE_CUDA = use_cuda

        self.lexicon = set(lexicon).union({"LIST_START", "LIST_END", "?"})

        # Calculate the maximum length
        self.maximumLength = 44 
        # self.maximumLength = max(len(l)
        #                          for t in tasks
        #                          for xs, y in self.tokenize(t.examples)
        #                          for l in [y] + [x for x in xs])

        super(
            LearnedFeatureExtractor,
            self).__init__(
            lexicon=list(
                self.lexicon),
            cuda=self.USE_CUDA,
            H=self.H,
            bidirectional=True)

#For ROBUSTFILL TASKS
class RobustFillLearnedFeatureExtractor(RecurrentFeatureExtractor):
    def tokenize(self, examples):
        return examples

    def __init__(self, lexicon, hidden=128, use_cuda=True): #was(self, tasks)
        self.lexicon = set(lexicon)
        self.USE_CUDA = use_cuda
        self.H = hidden

        super(RobustFillLearnedFeatureExtractor, self).__init__(lexicon=list(lexicon),
                                                        cuda=self.USE_CUDA,
                                                        H=self.H,
                                                        bidirectional=True)
        self.MAXINPUTS = 10


class SketchFeatureExtractor(RecurrentFeatureExtractor):
    def tokenize(self, sketch):
        return [([], sketch)] #I think this is what I want

    def __init__(self, lexicon, hidden=128, use_cuda=True): #was(self, tasks)
        self.lexicon = set(lexicon)
        self.USE_CUDA = use_cuda
        self.H = hidden

        super(SketchFeatureExtractor, self).__init__(lexicon=list(lexicon),
                                                        cuda=self.USE_CUDA,
                                                        H=self.H,
                                                        bidirectional=True)

class RegexFeatureExtractor(RecurrentFeatureExtractor):
    def tokenize(self, examples):
        #for example in examples:
        return [([], example) for example in examples]

    def __init__(self, lexicon, hidden=128, use_cuda=True): #was(self, tasks)
        self.lexicon = set(lexicon)
        self.USE_CUDA = use_cuda
        self.H = hidden

        super(RegexFeatureExtractor, self).__init__(lexicon=list(lexicon),
                                                        cuda=self.USE_CUDA,
                                                        H=self.H,
                                                        bidirectional=True)
        self.MAXINPUTS = 10

class AlgolispIOFeatureExtractor(RecurrentFeatureExtractor):
    def tokenize(self, IO):
        #for example in examples:
        return [([], tokenize_for_dc(example, digit_enc=self.digit_enc)) for example in IO]

    def __init__(self, lexicon, hidden=128, use_cuda=True, digit_enc=False): #was(self, tasks)
        self.lexicon = set(lexicon)
        self.USE_CUDA = use_cuda
        self.H = hidden
        self.digit_enc = digit_enc

        super(AlgolispIOFeatureExtractor, self).__init__(lexicon=list(lexicon),
                                                        cuda=self.USE_CUDA,
                                                        H=self.H,
                                                        bidirectional=True)
        self.MAXINPUTS = 10


class HoleSpecificFeatureExtractor(nn.Module):

    def __init__(self, exampleExtractor, sketchExtractor, hidden=128, use_cuda=True):
        super(HoleSpecificFeatureExtractor, self).__init__()

        self.H = hidden
        self.exampleExtractor = exampleExtractor
        self.sketchExtractor = sketchExtractor

        self.combining_layer = nn.Linear(self.exampleExtractor.outputDimensionality+self.sketchExtractor.outputDimensionality, self.H) # TODO

        if use_cuda:
            self.use_cuda = True
            self.cuda() # is this okay?

    def forward(self, rec_input):
        IO, sketch = rec_input

        example_features = self.exampleExtractor.featuresOfExamples(IO) #deal with tokenization and stuff in here
        sketch_features = self.sketchExtractor.featuresOfExamples(sketch) #deal with tokenization and stuff in here

        #print("example_features dims:", example_features.size())
        e = torch.cat((example_features, sketch_features),0) # TODO: does this make sense??

        e = F.relu(self.combining_layer(e)) 
        #print("output dims:", e.size())
        return e

    def featuresOfExamples(self, rec_input):
        return self(rec_input)

    @property
    def outputDimensionality(self): return self.H


class DeepcoderRecognitionModel(nn.Module):
    def __init__(
            self,
            featureExtractor,
            grammar,
            hidden=[128],
            activation="relu",
            cuda=False,
            contextual=False): #TODO implement this
        super(DeepcoderRecognitionModel, self).__init__()
        self.grammar = grammar
        self.use_cuda = cuda
        if cuda:
            self.cuda()

        self.featureExtractor = featureExtractor
        # Sanity check - make sure that all of the parameters of the
        # feature extractor were added to our parameters as well
        if hasattr(featureExtractor, 'parameters'):
            for parameter in featureExtractor.parameters():
                assert any(
                    myParameter is parameter for myParameter in self.parameters())

        self.hiddenLayers = []
        inputDimensionality = featureExtractor.outputDimensionality
        for h in hidden:
            layer = nn.Linear(inputDimensionality, h)
            if cuda:
                layer = layer.cuda()
            self.hiddenLayers.append(layer)
            inputDimensionality = h

        if activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "relu":
            self.activation = _relu
        elif activation == "tanh":
            self.activation = F.tanh
        else:
            raise Exception('Unknown activation function ' + str(activation))

        self.logVariable = nn.Linear(inputDimensionality, 1)
        self.logProductions = nn.Linear(inputDimensionality, len(self.grammar))
        if cuda:
            self.logVariable = self.logVariable.cuda()
            self.logProductions = self.logProductions.cuda()

        #add optimizer
        self.opt = torch.optim.Adam(
            self.parameters(), lr=0.0001, eps=1e-3, amsgrad=True)
        # if cuda:
        #     self.opt = self.opt.cuda()


        #super(RobustFill, self).cuda(*args, **kwargs)

    def productionEmbedding(self):
        # Calculates the embedding 's of the primitives based on the last
        # weight layer
        w = self.logProductions._parameters['weight']
        # w: len(self.grammar) x E
        if self.use_cuda:
            w = w.cpu()
        e = dict({p: w[j, :].data.numpy()
                  for j, (t, l, p) in enumerate(self.grammar.productions)})
        e[Index(0)] = w[0, :].data.numpy()
        return e


    def forward(self, features):
        for layer in self.hiddenLayers:
            features = self.activation(layer(features))
        h = features
        # added the squeeze
        return self.logVariable(h), self.logProductions(h)

    def _run(self, rec_input):
        # TODO, may want to do this 
        features = self.featureExtractor.featuresOfExamples(rec_input) #TODO LOOK AT THIS
        return self(features)

    def loss(self, rec_input, program, request):
        variables, productions = self._run(rec_input)
        g = Grammar(
            variables, [
                (productions[k].view(1), t, prog) for k, (_, t, prog) in enumerate(
                    self.grammar.productions)])
        return - g.logLikelihood(request, program)

    def optimizer_step(self, rec_input, program, request): #TODO: program shouldn't be full program, but thing that was hole
        #print("Warning, no batching yet")

        self.opt.zero_grad()


        loss = self.loss(rec_input, program, request) #can throw in a .mean() here when you are batching
        loss.backward()
        self.opt.step()

        return loss.data.item()

    def infer_grammar(self, rec_input):
        variables, productions = self._run(rec_input)
        g = Grammar(
            variables, [
                (productions[k].view(1), t, prog) for k, (_, t, prog) in enumerate(
                    self.grammar.productions)])
        return g



class ImprovedRecognitionModel(RecognitionModel): # TODO change name
    def __init__(
            self,
            featureExtractor,
            grammar,
            hidden=[128],
            activation="relu",
            cuda=False,
            contextual=False): #TODO implement this

        super(ImprovedRecognitionModel, self).__init__(
            featureExtractor,
            grammar,
            hidden=hidden,
            activation=activation,
            cuda=cuda,
            contextual=contextual)

        self.opt = torch.optim.Adam(
            self.parameters(), lr=0.0001, eps=1e-3, amsgrad=True)

        if cuda:
            self.cuda()
        

    def loss(self, rec_input, program, sketch, request):
        g = self.infer_grammar(rec_input)
        ll, _ = g.sketchLogLikelihood(request, program, sketch)
        #print(ll)
        return -ll


    def optimizer_step(self, rec_input, program, sketch, request):
        #print("Warning, no batching yet")
        if program == sketch: 
            print("prog == sketch, no opt")
            return 0.
        else:
            self.opt.zero_grad()
            loss = self.loss(rec_input, program, sketch, request) #can throw in a .mean() here when you are batching
            loss.backward()
            self.opt.step()
            return loss.data.item()

    def infer_grammar(self, rec_input):
        features = self.featureExtractor.featuresOfExamples(rec_input)
        g = self(features)
        return g



def load_rb_dc_model_from_path(path, max_length, max_index, improved_dc_grammar, cuda=True):
    basegrammar = Grammar.fromProductions(RobustFillProductions(max_length, max_index))

    if improved_dc_grammar:
        specExtractor = RobustFillLearnedFeatureExtractor(printable[:-4], hidden=128, use_cuda=cuda)
        vocab = robustfill_vocab(basegrammar)
        sketchExtractor = SketchFeatureExtractor(vocab, hidden=128, use_cuda=cuda)
        extractor = HoleSpecificFeatureExtractor(specExtractor, sketchExtractor, hidden=128, use_cuda=cuda)
        dcModel = ImprovedRecognitionModel(extractor, basegrammar, hidden=[128], cuda=cuda, contextual=False)
    else:
        extractor = RobustFillLearnedFeatureExtractor(printable[:-4], hidden=128)  # probably want to make it much deeper .... 
        dcModel = DeepcoderRecognitionModel(extractor, basegrammar, hidden=[128], cuda=True)  # probably want to make it much deeper .... 
    
    dcModel.load_state_dict(torch.load(path))
    return dcModel


if __name__ == '__main__':


    #Testing ROBUSTFILL:   
    from data_src.makeRegexData import sample_datum
    from util.regex_util import basegrammar, PregHole, regex_prior

    import pregex as pre

    regex_vocab = list(printable[:-4]) + \
    [pre.OPEN, pre.CLOSE, pre.String, pre.Concat, pre.Alt, pre.KleeneStar, pre.Plus, pre.Maybe, PregHole] + \
    regex_prior.character_classes

    d = None
    while not d:
        d = sample_datum(g=basegrammar,
            N=4,
            compute_sketches=True,
            top_k_sketches=100,
            inv_temp=1.0,
            reward_fn=None,
            sample_fn=None,
            dc_model=None)

    input_vocab = printable[:-4]
    exampleExtractor = RegexFeatureExtractor(input_vocab, hidden=128)
    sketchExtractor = SketchFeatureExtractor(regex_vocab, hidden=128)
    extractor = HoleSpecificFeatureExtractor(exampleExtractor, sketchExtractor, hidden=128, use_cuda=True)
    deepcoderModel = ImprovedRecognitionModel(extractor, basegrammar, hidden=[128], cuda=True, contextual=False)


    for i in range(400):
        score = deepcoderModel.optimizer_step((d.IO, d.sketchseq), d.p, d.sketch, d.tp) #WRONG!! shouldn't infer p, but infer

    print(d.p)
    print(d.IO)
    print(d.sketchseq)
    g = deepcoderModel.infer_grammar((d.IO, d.sketchseq))
    print(g)


