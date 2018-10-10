#play with results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dill
import time
import argparse
from p_solved_hack import hack_percent_solved_n_checked

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


mem_problem_list = ['results/dc_T45_0.25_800k_pypy.p', 'results/dc_T34_0.25_2M_pypy.p', 'results/dc_T45_0.25_2M_pypy.p']

def percent_solved_n_checked(results, n_checked):
	return sum(any(result.hit and result.n_checked <= n_checked for result in result_list) for result_list in results.values())/len(results)
	#speed this up!!!


def percent_solved_time(results, time):
	return sum(any(result.hit and result.time <= time for result in result_list) for result_list in results.values())/len(results)

def generalization_ratio(results, n_checked):
	return 0 if sum(any(result.hit and result.n_checked <= n_checked for result in result_list) for result_list in results.values()) <= 0 else sum(any(result.g_hit and result.n_checked <= n_checked for result in result_list) for result_list in results.values())/sum(any(result.hit and result.n_checked <= n_checked for result in result_list) for result_list in results.values())


def rload(resultsfile):
	with open(resultsfile, 'rb') as savefile:
		results = dill.load(savefile)
	return results


def plot_result(results=None, baseresults=None, resultsfile=None, basefile='results/prelim_results_dc_baseline__test50_1536770416.p', plot_time=True, model_path=None, robustfill=False, rnn=False):
	assert bool(results) != bool(resultsfile)  # xor of them
	if not results:
		with open(resultsfile, 'rb') as savefile:
			results = dill.load(savefile)
	if not baseresults:
		try:	
			with open(basefile, 'rb') as savefile:
				baseresults = dill.load(savefile)
		except FileNotFoundError:
			with open('../../'+basefile, 'rb') as savefile:  # in case it is not in this directory
				baseresults = dill.load(savefile)

	filename = model_path + str(time.time()) if model_path else str(time.time())
	baseline=basefile
	x_axis = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800, 900, 1000, 2000, 4000]  # TODO
	y_axis = [percent_solved_n_checked(results, x) for x in x_axis]
	base_y = [percent_solved_n_checked(baseresults, x) for x in x_axis]
	x_time = [0.1, 1, 1.5, 2, 2.5, 3, 4, 5, 10, 15, 20, 25, 30, 45, 60, 90, 120, 180, 200, 240]
	our_time = [percent_solved_time(results, x) for x in x_time]
	base_time = [percent_solved_time(baseresults, x) for x in x_time]
	print(x_axis)
	print(y_axis)
	fig, ax = plt.subplots()
	plt.plot(x_axis, y_axis, label='flexible neural sketch (ours)', linewidth=4.0, marker="o")
	if rnn: plt.plot(x_axis, base_y, label='rnn baseline', linewidth=4.0, marker="o")
	else: plt.plot(x_axis, base_y, label='deepcoder baseline', linewidth=4.0, marker="o")
	
	if robustfill:
		ax.set(title='Robustfill problems solved by neural sketch system', ylabel='% of problems solved', xlabel='Number of candidates evaluated per problem')
	else:
		ax.set(title='Deepcoder problems solved by neural sketch system', ylabel='% of problems solved', xlabel='Number of candidates evaluated per problem')
	ax.legend(loc='best')
	savefile='plots/' + baseline.split('/')[-1] + filename.split('/')[-1] + '.eps'
	plt.savefig(savefile)
	if plot_time:
		fig, ax = plt.subplots()
		plt.plot(x_time, our_time, label='flexible neural sketch (ours)', linewidth=4.0, marker="o")
		if rnn: plt.plot(x_time, base_time, label='rnn baseline', linewidth=4.0, marker="o")
		else: plt.plot(x_time, base_time, label='deepcoder baseline', linewidth=4.0, marker="o")

		if robustfill:
			ax.set(title='Robustfill problems solved by neural sketch system', ylabel='% of problems solved', xlabel='wall clock time (seconds)')
		else:
			ax.set(title='Deepcoder problems solved by neural sketch system', ylabel='% of problems solved', xlabel='wall clock time (seconds)')
		ax.legend(loc='best')
		savefile='plots/time_' + baseline.split('/')[-1] + filename.split('/')[-1] + '.eps'
		#savefile = 'plots/time_prelim.png'
		plt.savefig(savefile)


def plot_result_list(file_list, legend_list, filename, robustfill=False, plot_time=True, generalization=False, double=False, title='NA', max_budget=500000):
	result_list = []
	if double:
		l = len(file_list)
		fl = zip(file_list[:l],file_list[l:])
		for f1, f2 in file_list:
			with open(f1, 'rb') as savefile:
				r1 = dill.load(savefile)
			with open(f2, 'rb') as savefile:
				r2 = dill.load(savefile)
			result_list.append(r1+r2)
	else:
		for file in file_list:
			if file in mem_problem_list:
				result_list.append(file)
			else:
				with open(file, 'rb') as savefile:
					result_list.append(dill.load(savefile))
	if max_budget > 40000:
		x_axis= [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800, 900, 1000, 2000, 4000, 5000, 8000, 10000, 12000, 15000, 20000, 40000, 50000, 60000, 80000] + list(range(100000, max_budget, 100000)) # TODO
	else: 
		x_axis= [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800, 900, 1000, 2000, 4000, 5000, 8000, 10000, 12000, 15000, 20000, 40000] # TODO
	x_time = [0.1, 1, 1.5, 2, 2.5, 3, 4, 5, 10, 15, 20, 25, 30, 45, 60, 90, 120, 140, 150, 160, 180, 200, 240, 300] #, 400]+  list(range(500,2600,100))

	fig, ax = plt.subplots()
	#fig.set_size_inches(10,6)
	
	for result, legend in zip(result_list, legend_list):

		if 'RNN' in legend or 'RobustFill' in legend:
			y = percent_solved_n_checked(result, x_axis[-1])
			y_axis = [y for _ in x_axis]
			if '50' in legend:
				plt.semilogx(x_axis, y_axis, label=legend, linewidth=2.0, linestyle='--', c='C6')
			else:
				plt.semilogx(x_axis, y_axis, label=legend, linewidth=2.0, linestyle='--', c='C4')
		else:
			if generalization:
				y_axis = [generalization_ratio(result, x) for x in x_axis]
			else:
				if result in mem_problem_list:
					y_axis = [hack_percent_solved_n_checked(result, x) for x in x_axis] #TODO!!!!
				else:
					y_axis = [percent_solved_n_checked(result, x) for x in x_axis]
			if "Deepcoder" in legend: plt.plot(x_axis, y_axis, label=legend, linewidth=2.0, linestyle='-.', c='C3')	
			else: plt.semilogx(x_axis, y_axis, label=legend, linewidth=2.0)

	
	if robustfill:
		if generalization: ax.set(title='String transformation generalization_ratio', ylabel='num correct hits/num total hits', xlabel='Number of candidates evaluated per problem')
		else: 
			ax.set(title='String editing programs' if title=='NA' else title, ylabel='% of problems solved', xlabel='Number of candidates evaluated per problem')
			#ax.set_aspect(0.5)
	else:
		ax.set(title='length 3 test programs' if title=='NA' else title, ylabel='% of problems solved', xlabel='Number of candidates evaluated per problem')
		#ax.set(title='Trained on length 4 programs, tested on length 5 programs', ylabel='% of problems solved', xlabel='Number of candidates evaluated per problem')

	ax.legend(loc='best')
	savefile='plots/' +filename+ '.eps'
	plt.savefig(savefile)
	

	if plot_time:

		fig, ax = plt.subplots()

		for result, legend in zip(result_list, legend_list):
			our_time = [percent_solved_time(result, x) for x in x_time]
			if 'RNN' in legend or 'RobustFill' in legend:
				if '50' in legend:
					plt.plot(x_time, our_time, label=legend, linewidth=2.0, linestyle='--', c='C6')
				else:
					plt.plot(x_time, our_time, label=legend, linewidth=2.0, linestyle='--', c='C4')
			elif "Deepcoder" in legend:
				plt.plot(x_time, our_time, label=legend, linewidth=2.0, linestyle='-.', c='C3')
			else:
				plt.plot(x_time, our_time, label=legend, linewidth=2.0)

		if robustfill:
			ax.set(title='String editing problems - evaluation time', ylabel='% of problems solved', xlabel='wall clock time (seconds)')
		else:
			ax.set(title='List processing problems solved by neural sketch system', ylabel='% of problems solved', xlabel='wall clock time (seconds)')
		ax.legend(loc='best')

		savefile='plots/time_' + filename + '.eps'
		#savefile = 'plots/time_prelim.png'
		plt.savefig(savefile)


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--basefile', type=str, default='results/prelim_results_dc_baseline__test50_1536770416.p')
	parser.add_argument('--resultsfile', type=str, default='results/prelim_results_wdcModel__test50_1536803560.p')
	parser.add_argument('--rb', action='store_true')
	parser.add_argument('--rnnbase', action='store_true')
	args = parser.parse_args()
	# filename = 'results/prelim_results_dc_baseline__test20_1536185296.p'
	# #filename = 'results/prelim_results_wdcModel__test20_1536184791.p' no input, i borked it
	# filename = 'results/prelim_results_wdcModel__test20_1536102640.p'
	# filename = 'results/prelim_results_dc_baseline__test200_1536244251.p'
	# filename = 'results/prelim_results_wdcModel__test50_1536249076.p'
	# filename = 'results/prelim_results_dc_baseline__test50_1536246985.p'
	#baseline = 'results/prelim_results_dc_baseline__test50_1536260203.p'
	#filename = 'results/prelim_results_wdcModel__test50_1536249614.p'
	#baseline = 'results/prelim_results_dc_baseline__test50_1536770416.p'
	#filename = 'results/prelim_results_wdcModel__test50_1536770193.p' # supervised trained
	#filename = 'results/prelim_results_wdcModel__test50_1536770602.p' # rl trained old version
	#filename = 'results/prelim_results_wdcModel__test50_1536818453.p' # exp86 without var red
	#filename = 'results/prelim_results_wdcModel__test50_1536819663.p' # linear 8 with variance reduction
	#filename = 'results/prelim_results_wdcModel__test50_1536854115.p' #flat without var red
	#filename = 'results/prelim_results_wdcModel__test50_1536854557.p' # exp86 with variance red
	#filename = 'results/prelim_results_wdcModel__test50_1536803560.p' # exp86 w no var red take 2

	plot_result(resultsfile=args.resultsfile, basefile=args.basefile, plot_time=True, robustfill=args.rb, rnn=args.rnnbase)

