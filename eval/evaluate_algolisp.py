#evaluate.py
import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./ec'))

#import statements
import argparse
import torch
from torch import nn, optim
import random
import math
import time
import pickle
import dill
import copy

from util.algolisp_pypy_util import AlgolispResult, SketchTup, alternate, pypy_enumerate, algolisp_enumerate #TODO
from util.algolisp_util import tokenize_for_robustfill, seq_to_tree, tree_to_prog, tree_depth, tokenize_IO_for_robustfill  # TODO
from data_src.makeAlgolispData import batchloader, basegrammar
from train.algolisp_train_dc_model import newDcModel

from plot.manipulate_results import percent_solved_n_checked, percent_solved_time, plot_result, solve_time_percentile

from grammar import Grammar
from itertools import islice

from torch.multiprocessing import Pool, Queue, Process
import functools

from task import Task, EvaluationTimeout
import signal

from memory_profiler import profile
import traceback
#train & use dcModel
#which requires converting programs to EC domain
parser = argparse.ArgumentParser()
parser.add_argument('--n_test', type=int, default=9967) #only_passable length is 9807 for dev, 8903 for eval, total eval is 9967
parser.add_argument('--dcModel', action='store_true', default=True)
parser.add_argument('--dcModel_path',type=str, default="./saved_models/algolisp_dc_model.p")
parser.add_argument('--improved_dc_grammar', action='store_true', default=True)
parser.add_argument('--dc_baseline', action='store_true')
parser.add_argument('--n_samples', type=int, default=10)
parser.add_argument('--mdl', type=int, default=14)  #9
parser.add_argument('--n_examples', type=int, default=5)
parser.add_argument('--model_path', type=str, default="./saved_models/algolisp_holes.p")
parser.add_argument('--max_to_check', type=int, default=5000)
parser.add_argument('--resultsfile', type=str, default='first_rnn_algolisp_results_holes')
parser.add_argument('--shuffled', action='store_true')
parser.add_argument('--beam', action='store_true', default=True)
parser.add_argument('--dataset', type=str, default='eval')
parser.add_argument('--pypy', action='store_true')
parser.add_argument('--parallel', action='store_true', default=True)
parser.add_argument('--chunksize', type=int, default=100)
parser.add_argument('--n_processes', type=int, default=48)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--queue', action='store_true')
parser.add_argument('--only_passable', action='store_true')
parser.add_argument('--filter_depth', nargs='+', type=int, default=None)
parser.add_argument('--timeout', type=int, default=None)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--IO2seq', action='store_true')
parser.add_argument('--odd', action='store_true')
parser.add_argument('--even', action='store_true')
parser.add_argument('--geq', action='store_true')
parser.add_argument('--leq', action='store_true')
parser.add_argument('--lt', action='store_true')
parser.add_argument('--gt', action='store_true')
parser.add_argument('--n_split', default=None, type=int)
parser.add_argument('--start_at_debug', default=None, type=int)
parser.add_argument('--digit_enc', action='store_true')
args = parser.parse_args()

assert not (args.even and args.odd)
args.cpu = args.cpu or args.parallel #if parallel, then it must be cpu only
nSamples = args.n_samples
mdl = args.mdl
nExamples = args.n_examples #TODO
max_to_check = args.max_to_check
improved_dc_grammar = args.improved_dc_grammar
if args.dcModel: improved_dc_grammar = True
#if improved_dc_grammar: assert args.dcModel

#args.chunksize=args.n_test/args.n_processes

def untorch(g):
	if type(g.logVariable) == float:
		return g
	else:
		return Grammar(g.logVariable.data.tolist()[0], 
                               [ (l.data.tolist()[0], t, p)
                                 for l, t, p in g.productions])


#@profile
def evaluate_datum(i, datum, model, dcModel, nRepeats, mdl, max_to_check, timeout=None):
	try:
		if timeout is not None:
			#print("hit set timeout")
			def timeoutCallBack(_1, _2): raise EvaluationTimeout()
			#signal.signal(signal.SIGVTALRM, timeoutCallBack)
			#signal.setitimer(signal.ITIMER_REAL, timeout)
			signal.signal(signal.SIGALRM, timeoutCallBack)
			signal.alarm(timeout)

		t = time.time()
		results = []
		print("eval datum number", i)
		samples = {('<HOLE>',)}  # make more general #  TODO, i don't think 
		n_checked, n_hit = 0, 0
		if model:
			#print("num threads:", torch.get_num_threads())
			torch.set_num_threads(1)
			tnet = time.time()
			#print("task", i, "started using rnn")
			spec = tokenize_for_robustfill([datum.spec]) if not args.IO2seq else tokenize_IO_for_robustfill([datum.IO], digit_enc=args.digit_enc)
			if args.beam:
				samples, _scores = model.beam_decode(spec, beam_size=nRepeats)
			else:
				samples, _scores, _ = model.sampleAndScore(spec, nRepeats=nRepeats)

			#print("task", i, "done with rnn, took", time.time()-t, "seconds", flush=True)
			# only loop over unique samples:
			samples = {tuple(sample) for sample in samples}  # only 
		if (not improved_dc_grammar) or (not dcModel):
			g = basegrammar if not dcModel else dcModel.infer_grammar((datum.spec if not args.IO2seq else datum.IO))  # TODO pp
			g = untorch(g)
		sketchtups = []
		for sample in samples:
			try:
				tr = seq_to_tree(sample)
				#print(tr)
				sk = tree_to_prog(tr)

			except EvaluationTimeout:
				raise EvaluationTimeout()

			except Exception as e: # TODO: needs to be fixed
				traceback.clear_frames(e.__traceback__)
				print("EXCEPTION IN PARSE:,", e)
				#traceback.print_tb(e.__traceback__)
				del e
				n_checked += 1
				results.append( AlgolispResult(sample, None, False, n_checked, time.time()-t) )
				continue

			if improved_dc_grammar:
				g = untorch(dcModel.infer_grammar(( (datum.spec if not args.IO2seq else datum.IO), sample))) 

			sketchtups.append(SketchTup(sk, g))

		# only loop over unique sketches:
		sketchtups = {sk for sk in sketchtups} #fine
		#print("sketchtups:", sketchtups)

		#alternate which sketch to enumerate from each time
		#print("task", i, "starting enum outer, took", time.time()-t, "seconds", flush=True)
		if args.pypy:
			enum_results, n_checked, n_hit = pypy_enumerate(datum.tp, datum.IO, datum.schema_args, mdl, sketchtups, n_checked, n_hit, t, max_to_check, i)
		else:
			enum_results, n_checked, n_hit = algolisp_enumerate(datum.tp, datum.IO, datum.schema_args, mdl, sketchtups, n_checked, n_hit, t, max_to_check, i) #might need more than IO
		del sketchtups
		#del g
		if model:
			del model
		if dcModel:
			del dcModel
		return results + enum_results

	except KeyError as e:
		print("on task", i, "KeyError:", e)
		return [AlgolispResult(None, None, False, n_checked, time.time()-t)]

	except EvaluationTimeout:
		print("Timed out while evaluating task", i)
		return [AlgolispResult(None, None, False, n_checked, time.time()-t)]

	finally:
		######TODO: want search time and total time to hit task ######
		if timeout is not None:
			#signal.signal(signal.SIGVTALRM, lambda *_: None)
			#signal.setitimer(signal.ITIMER_REAL, 0)
			signal.alarm(0)
		print(f"task {i}:")
		print(f"evaluation for task {i} took {time.time()-t} seconds")
		print(f"For task {i}, tried {n_checked} candidates, found {n_hit} hits", flush=True)



def evaluate_dataset(model, dataset, nRepeats, mdl, max_to_check, dcModel=None):
	t = time.time()
	if model is None:
		print("evaluating dcModel baseline")

	print("num threads outer:", torch.get_num_threads())
	if args.parallel:
		if args.queue:
			print("using queue ...")
			def consumer(inQ, outQ):
				while True:
					try:
						# get a new message
						val = inQ.get()
						# this is the 'TERM' signal
						if val is None:
						    break;
						# process the data
						ret = f(val)

						if args.debug:
							from pympler import summary
							from pympler import muppy
							all_objects = muppy.get_objects()
							sum1 = summary.summarize(all_objects)
							print("summary:")
							summary.print_(sum1)
							from pympler import tracker
							tr = tracker.SummaryTracker()
							print("diff:")
							tr.print_diff()

						outQ.put( ret )
					except Exception as e:
						print("error!", e)
						break

			def process_data(data_list, inQ, outQ):
				# send pos/data to workers
				for i, dat in enumerate(data_list):
					inQ.put(dat)
				# process results
				res = []
				# for _ in range(args.n_test):
				# 	res.append( outQ.get() )
				# return res
				return [outQ.get() for _ in range(i+1)]

			inQ = Queue()
			outQ = Queue()
			# instantiate workers

			# if not args.n_queues:
			# queue_size = int(args.n_test/args.n_queues)
			# resuts_list = [] 
			# for x in range(n_queues):

			workers = [Process(target=consumer, args=(inQ, outQ))
			           for i in range(args.n_processes)]
			# start the workers
			for w in workers:
			    w.start()
			# gather some data
			data_list = enumerate(dataset)
			results_list = process_data(data_list, inQ, outQ) #can just also have length of list
			# tell all workers, no more data (one msg for each)
			for i in range(args.n_processes):
			    inQ.put(None)
			# join on the workers
			for w in workers:
			    w.join()
		else:
			#f = lambda i, datum: (datum, list(evaluate_datum(i, datum, model, dcModel, nRepeats, mdl, max_to_check)))
			with Pool(processes=args.n_processes) as pool:
				print("started pool...")
				results_list = pool.map(f, enumerate(dataset), chunksize=args.chunksize) #imap_unordered
	else:
		#result_dict = {datum: list(evaluate_datum(i, datum, model, dcModel, nRepeats, mdl, max_to_check) ) for i, datum in enumerate(dataset)}
		results_list = map(f, enumerate(dataset))

	result_dict = {d: res_list for d, res_list in results_list}		
	return result_dict

def save_results(results, args):
	timestr = str(int(time.time()))
	r = '_test' + str(args.n_test) + '_'
	if args.resultsfile != 'NA':
		filename = 'algolisp_results/' + args.resultsfile + '.p'
	elif args.dc_baseline:
		filename = "algolisp_results/algolisp_prelim_results_dc_baseline_" + r + timestr + '.p'
	elif args.pretrained:
		filename = "algolisp_results/algolisp_prelim_results_rnn_baseline_" + r + timestr + '.p'
	else:
		dc = 'wdcModel_' if args.dcModel else ''
		filename = "algolisp_results/algolisp_prelim_results_" + dc + r + timestr + '.p'
	with open(filename, 'wb') as savefile:
		dill.dump(results, savefile)
		print("results file saved at", filename)
	return savefile

if __name__=='__main__':
	#load the model
	if args.dc_baseline:
		print("computing dc baseline, no model")
		assert args.dcModel
		model = None
	else:
		print("loading model with holes")
		if args.cpu: 
			model = torch.load(args.model_path, map_location=lambda storage, loc: storage)
		else:
			model = torch.load(args.model_path) #TODO
			model.cuda()
	if args.dcModel:
		print("loading dcModel")
		
		if args.cpu:
			dcModel=newDcModel(cuda=False, IO2seq=args.IO2seq, digit_enc=args.digit_enc)
			dcModel.load_state_dict(torch.load(args.dcModel_path, map_location=lambda storage, loc: storage))
		else:
			dcModel=newDcModel(IO2seq=args.IO2seq, digit_enc=args.digit_enc)
			dcModel.load_state_dict(torch.load(args.dcModel_path))
			dcModel.cuda()
	else: dcModel = None

	###load the test dataset###
	if args.odd:
		include_only = [ ["lambda1", ["==", ["%", "arg1", "2"], "1"]] ]
	elif args.even: 
		include_only = [ ["lambda1", ["==", ["%", "arg1", "2"], "0"]] ] 
	elif args.leq:
		include_only = ["<="]
	elif args.geq:
		include_only = [">="]
	elif args.gt:
		include_only = [">"]
	elif args.lt:
		include_only = ["<"]
	else: 
		include_only = None

	full_dataset = batchloader(args.dataset,
							batchsize=1,
                            compute_sketches=False,
                            dc_model=None,
                            improved_dc_model=False,
                            only_passable=args.only_passable,
                            filter_depth=args.filter_depth,
                            include_only=include_only) #TODO
	

	# from collections import Counter

	if args.start_at_debug:
		full_dataset = islice(full_dataset, args.start_at_debug, args.n_test)
	# c = Counter()
	# c.update(tree_depth(seq_to_tree(d.pseq)) for d in dataset)
	# print(c)
	# assert False


	#this needs to be here for stuped reasons ...
	def f(i_datum):
		f_partial = functools.partial(evaluate_datum, model=model, dcModel=dcModel, nRepeats=nSamples, mdl=mdl, max_to_check=max_to_check, timeout=args.timeout)
		i, datum = i_datum
		return (datum, list(f_partial(i,datum)))

	if args.n_split:
		cutsize = int(args.n_test/args.n_split)
		results = {}
		for i in range(0,  args.n_test, cutsize):
			dataset = islice(full_dataset, i, i+cutsize)
			results.update(evaluate_dataset(model, dataset, nSamples, mdl, max_to_check, dcModel=dcModel) )

	else:
		dataset = islice(full_dataset, args.n_test)
		results = evaluate_dataset(model, dataset, nSamples, mdl, max_to_check, dcModel=dcModel)

	# count hits
	hits = sum(any(result.hit for result in result_list) for result_list in results.values())
	print(f"hits: {hits} out of {len(results)}, or {100*hits/len(results)}% accuracy")

	# I want a plot of the form: %solved vs n_hits
	# x_axis = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800, 900, 1000, 2000, 4000]  # TODO
	# y_axis = [percent_solved_n_checked(results, x) for x in x_axis]

	# print("percent solved vs number of evaluated programs")
	# print("num_checked:", x_axis)
	# print("num_solved:", y_axis)

	file = save_results(results, args)

	print("25th percentile solve time:")
	print(solve_time_percentile(results, 25, use_misses=False))

	print("median solve time:")
	print(solve_time_percentile(results, 50, use_misses=False))

	print("75th percentile solve time:")
	print(solve_time_percentile(results, 75, use_misses=False))

	print("FILTERED:")
	filt_list = list(islice(batchloader(args.dataset,
											batchsize=1,
											compute_sketches=False,
											only_passable=True,
											include_only=include_only), args.n_test))
	filtered_results = {key: val for key, val in results.items() if any(hash(key) == hash(filt) for filt in filt_list)}

	filtered_hits = sum(any(result.hit for result in result_list) for result_list in filtered_results.values())


	from util.algolisp_pypy_util import test_program_on_IO
	from program_synthesis.algolisp.dataset import executor

	executor_ = executor.LispExecutor()
	filtered_results = {key: val for key, val in results.items() if test_program_on_IO(key.p.evaluate([]), key.IO, key.schema_args, executor_)}
	print(f"hits: {filtered_hits} out of {len(filtered_results)}, or {100*filtered_hits/len(filtered_results)}% accuracy")

	file = save_results(results, args)

	print("25th percentile solve time:")
	print(solve_time_percentile(filtered_results, 25, use_misses=False))

	print("median solve time:")
	print(solve_time_percentile(filtered_results, 50, use_misses=False))

	print("75th percentile solve time:")
	print(solve_time_percentile(filtered_results, 75, use_misses=False))

	#plot_result(results=results, plot_time=True, model_path=args.model_path) #doesn't account for changing result thingy

