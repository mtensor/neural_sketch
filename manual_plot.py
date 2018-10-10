#manual_plot.py

from manipulate_results import *

filename = 
file_list = 
dc = 
rnn = 


def load_result(filename):
	with open(file, 'rb') as savefile:
		result = dill.load(savefile)
	return result

x_axis= [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800, 900, 1000, 2000, 4000, 5000, 8000, 10000, 12000, 15000, 20000, 40000] #, 50000, 60000, 80000, 100000] # TODO
x_time = [0.1, 1, 1.5, 2, 2.5, 3, 4, 5, 10, 15, 20, 25, 30, 45, 60, 90, 120, 140, 150, 160, 180, 200, 240, 300, 400, 500, 600, 800, 1200]
fig, ax = plt.subplots()


result = load_result()
y_axis = [percent_solved_n_checked(result, x) for x in x_axis]
plt.plot(x_axis, y_axis, label=legend, linewidth=4.0)


result = load_result()
y_axis = [percent_solved_n_checked(result, x) for x in x_axis]
plt.plot(x_axis, y_axis, label=legend, linewidth=4.0)


result = load_result()
y_axis = [percent_solved_n_checked(result, x) for x in x_axis]
plt.plot(x_axis, y_axis, label=legend, linewidth=4.0)




ax.set(title='String transformation generalization_ratio', ylabel='num correct hits/num total hits', xlabel='Number of candidates evaluated per problem')
#ax.set(title='List processing problems solved by neural sketch system', ylabel='% of problems solved', xlabel='Number of candidates evaluated per problem')
ax.legend(loc='best')
savefile='plots/' +filename+ '.eps'
plt.savefig(savefile)


# if plot_time:

# 	fig, ax = plt.subplots()

# 	for result, legend in zip(result_list, legend_list):
# 		our_time = [percent_solved_time(result, x) for x in x_time]
# 		plt.plot(x_time, our_time, label=legend, linewidth=4.0)

# 	if robustfill:
# 		ax.set(title='String transformation problems solved by neural sketch system', ylabel='% of problems solved', xlabel='wall clock time (seconds)')
# 	else:
# 		ax.set(title='List processing problems solved by neural sketch system', ylabel='% of problems solved', xlabel='wall clock time (seconds)')
# 	ax.legend(loc='best')

# 	savefile='plots/time_' + filename + '.eps'
# 	#savefile = 'plots/time_prelim.png'
# 	plt.savefig(savefile)


