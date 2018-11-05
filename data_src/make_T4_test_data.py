#make_T4_test_data.py
import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./ec'))

from eval.evaluate_deepcoder import *
from itertools import islice

##load the test dataset###
# test_data = ['data/DeepCoder_test_data/T3_A2_V512_L10_P500.txt']
# test_data = ['data/DeepCoder_test_data/T5_A2_V512_L10_P100_test.txt'] #modified from original
# test_data = ['data/DeepCoder_data/T3_A2_V512_L10_validation_perm.txt']

# test_data = ['data/DeepCoder_data/T4_A2_V512_L10_train_perm.txt']

test_data = ['data/DeepCoder_data/T44_test.txt']
# test_data = ['data/DeepCoder_data/T44_train.txt']

#test_data = ['data/DeepCoder_data/T2_A2_V512_L10_test_perm.txt']
dataset = batchloader(test_data, batchsize=1, N=5, V=Vrange, L=10, compute_sketches=False)
dataset = list(islice(dataset, 500))
with open('data/T44_test.p', 'wb') as savefile:
	pickle.dump(dataset, savefile)
	print("test file saved")