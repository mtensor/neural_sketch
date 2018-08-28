#deepcodermodel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimization
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np


import sys
sys.path.append("/om/user/mnye/ec")
#from recognition import RecurrentFeatureExtractor, RecognitionModel
from grammar import Grammar  #?

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

        x = variable(es, cuda=self.use_cuda)
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














class DeepcoderRecognitionModel(nn.Module):
    def __init__(
            self,
            featureExtractor,
            grammar,
            hidden=[128],
            activation="relu",
            cuda=False):
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

    # def taskEmbeddings(self, tasks):
    #     return {task: self.featureExtractor.featuresOfTask(task).data.numpy()
    #             for task in tasks}

    def forward(self, features):
        for layer in self.hiddenLayers:
            features = self.activation(layer(features))
        h = features
        # added the squeeze
        return self.logVariable(h), self.logProductions(h)

    # def frontierKL(self, frontier):
    #     features = self.featureExtractor.featuresOfTask(frontier.task)
    #     variables, productions = self(features)
    #     g = Grammar(
    #         variables, [
    #             (productions[k].view(1), t, program) for k, (_, t, program) in enumerate(
    #                 self.grammar.productions)])
    #     # Monte Carlo estimate: draw a sample from the frontier
    #     entry = frontier.sample()
    #     return - entry.program.logLikelihood(g)


    def _run(self, IO):
        #task = self.IO_to_task(IO) #TODO
        features = self.featureExtractor.featuresOfExamples(IO) #TODO LOOK AT THIS
        return self(features)


    def loss(self, IO, program, request):
        variables, productions = self._run(IO)
        g = Grammar(
            variables, [
                (productions[k].view(1), t, prog) for k, (_, t, prog) in enumerate(
                    self.grammar.productions)])
        print("WARNING: loss mode = dreamcoder")
        return - g.logLikelihood(request, program) #program.logLikelihood(g)


    def optimizer_step(self, IO, program, request):
        print("Warning, no batching yet")

        self.opt.zero_grad()


        loss = self.loss(IO, program, request) #can throw in a .mean() here when you are batching
        loss.backward()
        self.opt.step()

        return loss.data.item()


    def infer_grammar(self, IO):
        variables, productions = self._run(IO)
        g = Grammar(
            variables, [
                (productions[k].view(1), t, prog) for k, (_, t, prog) in enumerate(
                    self.grammar.productions)])
        return g



if __name__ == '__main__':
    from main_supervised_deepcoder import grammar, getInstance
    deepcoder_io_vocab = list(range(-128,128))

    inst = getInstance() #k_shot=4, max_length=30, verbose=False, with_holes=False, k=None)


    IO = inst['IO']
    print("IO:", IO)
    program = inst['p']
    print("program:", program)

    request = inst['tp']

    extractor = LearnedFeatureExtractor(deepcoder_io_vocab, hidden=128)
    deepcoderModel = DeepcoderRecognitionModel(extractor, grammar, hidden=[128], cuda=True)
    for i in range(400):
        score = deepcoderModel.optimizer_step(IO, program, request)

    g = deepcoderModel.infer_grammar(IO)



    # def replaceProgramsWithLikelihoodSummaries(self, frontier):
    #     return Frontier(
    #         [
    #             FrontierEntry(
    #                 program=self.grammar.closedLikelihoodSummary(
    #                     frontier.task.request,
    #                     e.program),
    #                 logLikelihood=e.logLikelihood,
    #                 logPrior=e.logPrior) for e in frontier],
    #         task=frontier.task)

    # def train(self, frontiers, _=None, steps=250, lr=0.0001, topK=1, CPUs=1,
    #           timeout=None, helmholtzRatio=0., helmholtzBatch=5000):
    #     """
    #     helmholtzRatio: What fraction of the training data should be forward samples from the generative model?
    #     """
    #     requests = [frontier.task.request for frontier in frontiers]
    #     frontiers = [frontier.topK(topK).normalize()
    #                  for frontier in frontiers if not frontier.empty]

    #     # We replace each program in the frontier with its likelihoodSummary
    #     # This is because calculating likelihood summaries requires juggling types
    #     # And type stuff is expensive!
    #     frontiers = [self.replaceProgramsWithLikelihoodSummaries(f).normalize()
    #                  for f in frontiers]

    #     # Not sure why this ever happens
    #     if helmholtzRatio is None:
    #         helmholtzRatio = 0.

    #     eprint("Training a recognition model from %d frontiers, %d%% Helmholtz, feature extractor %s." % (
    #         len(frontiers), int(helmholtzRatio * 100), self.featureExtractor.__class__.__name__))

    #     # The number of Helmholtz samples that we generate at once
    #     # Should only affect performance and shouldn't affect anything else
    #     HELMHOLTZBATCH = helmholtzBatch
    #     helmholtzSamples = []

    #     optimizer = torch.optim.Adam(
    #         self.parameters(), lr=lr, eps=1e-3, amsgrad=True)
    #     if timeout:
    #         start = time.time()

    #     with timing("Trained recognition model"):
    #         for i in range(1, steps + 1):
    #             if timeout and time.time() - start > timeout:
    #                 break
    #             losses = []

    #             if helmholtzRatio < 1.:
    #                 permutedFrontiers = list(frontiers)
    #                 random.shuffle(permutedFrontiers)
    #             else:
    #                 permutedFrontiers = [None]
    #             for frontier in permutedFrontiers:
    #                 # Randomly decide whether to sample from the generative
    #                 # model
    #                 doingHelmholtz = random.random() < helmholtzRatio
    #                 if doingHelmholtz:
    #                     if helmholtzSamples == []:
    #                         helmholtzSamples = \
    #                             list(self.sampleManyHelmholtz(requests,
    #                                                           HELMHOLTZBATCH,
    #                                                           CPUs))
    #                     if len(helmholtzSamples) == 0:
    #                         eprint(
    #                             "WARNING: Could not generate any Helmholtz samples. Disabling Helmholtz.")
    #                         helmholtzRatio = 0.
    #                         doingHelmholtz = False
    #                     else:
    #                         attempt = helmholtzSamples.pop()
    #                         if attempt is not None:
    #                             self.zero_grad()
    #                             loss = self.frontierKL(attempt)
    #                         else:
    #                             doingHelmholtz = False
    #                 if not doingHelmholtz:
    #                     if helmholtzRatio < 1.:
    #                         self.zero_grad()
    #                         loss = self.frontierKL(frontier)
    #                     else:
    #                         # Refuse to train on the frontiers
    #                         continue

    #                 if is_torch_invalid(loss):
    #                     if doingHelmholtz:
    #                         eprint("Invalid real-data loss!")
    #                     else:
    #                         eprint("Invalid Helmholtz loss!")
    #                 else:
    #                     loss.backward()
    #                     optimizer.step()
    #                     losses.append(loss.data.tolist()[0])
    #                     if False:
    #                         if doingHelmholtz:
    #                             eprint(
    #                                 "\tHelmholtz data point loss:",
    #                                 loss.data.tolist()[0])
    #                         else:
    #                             eprint(
    #                                 "\tReal data point loss:",
    #                                 loss.data.tolist()[0])
    #             if (i == 1 or i % 10 == 0) and losses:
    #                 eprint("Epoch", i, "Loss", sum(losses) / len(losses))
    #                 gc.collect()

    # def sampleHelmholtz(self, requests, statusUpdate=None, seed=None):
    #     if seed is not None:
    #         random.seed(seed)
    #     request = random.choice(requests)

    #     #eprint("About to draw a sample")
    #     program = self.grammar.sample(request, maximumDepth=6, maxAttempts=100)
    #     #eprint("sample", program)
    #     if program is None:
    #         return None
    #     task = self.featureExtractor.taskOfProgram(program, request)
    #     #eprint("extracted features")

    #     if statusUpdate is not None:
    #         # eprint(statusUpdate, end='')
    #         flushEverything()
    #     if task is None:
    #         return None

    #     if hasattr(self.featureExtractor, 'lexicon'):
    #         #eprint("tokenizing...")
    #         if self.featureExtractor.tokenize(task.examples) is None:
    #             return None

    #     frontier = Frontier([FrontierEntry(program=program,
    #                                        logLikelihood=0., logPrior=0.)],
    #                         task=task)
    #     #eprint("replacing with likelihood summary")
    #     frontier = self.replaceProgramsWithLikelihoodSummaries(frontier)
    #     #eprint("successfully got a sample")
    #     return frontier

    # def sampleManyHelmholtz(self, requests, N, CPUs):
    #     eprint("Sampling %d programs from the prior on %d CPUs..." % (N, CPUs))
    #     flushEverything()
    #     frequency = N / 50
    #     startingSeed = random.random()
    #     samples = parallelMap(
    #         1,
    #         lambda n: self.sampleHelmholtz(requests,
    #                                        statusUpdate='.' if n % frequency == 0 else None,
    #                                        seed=startingSeed + n),
    #         range(N))
    #     eprint()
    #     flushEverything()
    #     samples = [z for z in samples if z is not None]
    #     eprint()
    #     eprint("Got %d/%d valid samples." % (len(samples), N))
    #     flushEverything()

    #     return samples

    # def enumerateFrontiers(self,
    #                        tasks,
    #                        likelihoodModel,
    #                        solver=None,
    #                        enumerationTimeout=None,
    #                        testing=False,
    #                        CPUs=1,
    #                        frontierSize=None,
    #                        maximumFrontier=None,
    #                        evaluationTimeout=None):
    #     with timing("Evaluated recognition model"):
    #         grammars = {}
    #         for task in tasks:
    #             features = self.featureExtractor.featuresOfTask(task)
    #             variables, productions = self(features)
    #             # eprint("variable")
    #             # eprint(variables.data[0])
    #             # for k in range(len(self.grammar.productions)):
    #             #     eprint("production",productions.data[k])
    #             grammars[task] = Grammar(
    #                 variables.data.tolist()[0], [
    #                     (productions.data.tolist()[k], t, p) for k, (_, t, p) in enumerate(
    #                         self.grammar.productions)])

    #     return multicoreEnumeration(grammars, tasks, likelihoodModel,
    #                                 solver=solver,
    #                                 testing=testing,
    #                                 enumerationTimeout=enumerationTimeout,
    #                                 CPUs=CPUs, maximumFrontier=maximumFrontier,
    #                                 evaluationTimeout=evaluationTimeout)