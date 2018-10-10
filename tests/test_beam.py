#test_beam.py
from eval.evaluate_deepcoder import *

path = 'experiments/deepcoder_timeout_0.5_1537326548865/deepcoder_holes.p'

model = torch.load(path)

with open(args.precomputed_data_file, 'rb') as datafile:
	dataset = pickle.load(datafile)

print("loaded model and dataset")


datum = dataset[0]

print("IO:")
print(datum.IO)

tokenized = tokenize_for_robustfill([datum.IO])
#samples, _scores, _ = model.beam_decode(tokenized, nRepeats=nRepeats)

# samples, _scores, _ = model.sampleAndScore(tokenized, nRepeats=10)
# print(samples, _scores)
# assert False 
targets, scores = model.beam_decode(tokenized, beam_size=10, vocab_filter=None)



print(targets, scores)