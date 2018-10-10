#make frontier plot
from manipulate_results import *

resultsfile_dict = {5:'results/dc_timeout_0.25_40k_beam5.p', 20:'results/dc_timeout_0.25_40k_beam20.p', 50:'results/dc_timeout_0.25_40k.p', 100:'results/dc_timeout_0.25_40k_beam100.p'  } #fill this in 

result_dict = {}
for num, resultsfile in resultsfile_dict.items():
	with open(resultsfile, 'rb') as savefile:
		results = dill.load(savefile)
	result_dict[num] = results


def find_x_percent(result, x, n_checked_min=0, n_checked_max=40000):

	fudge = 0.005
	n_fudge = 3

	#n_checked_min = 0
	#n_checked_max = 40000

	#percent_min = percent_solved_n_checked(results, n_checked_min)
	#percent_max = percent_solved_n_checked(results, n_checked_max)

	n_checked_mid = int(n_checked_min + n_checked_max)/2

	if n_checked_mid > 39999: return 40000
	elif n_checked_mid < 2: return 1 

	percent_mid = percent_solved_n_checked(result, n_checked_mid)
	print("testing", n_checked_mid)
	print(n_checked_mid, "had percent of", percent_mid)

	if abs(x - percent_mid) < fudge: return n_checked_mid
	elif abs(n_checked_mid - n_checked_min) < n_fudge or abs(n_checked_mid - n_checked_max) < n_fudge: return n_checked_mid 
	elif x > percent_mid: return find_x_percent(result, x, n_checked_min=n_checked_mid, n_checked_max=n_checked_max)
	elif x < percent_mid: return find_x_percent(result, x, n_checked_min=n_checked_min, n_checked_max=n_checked_mid)







line_60 = list(zip(*[(find_x_percent(result, .850), beam_size) for beam_size, result in result_dict.items()]))

line_40 = list(zip(*[(find_x_percent(result, .80), beam_size) for beam_size, result in result_dict.items()]))

line_20 = list(zip(*[(find_x_percent(result, .70), beam_size) for beam_size, result in result_dict.items()]))


result_list = [line_20, line_40, line_60]
legend_list = ["70% solved", "80% solved", "85% solved"]



#plot the 3 lines:

fig, ax = plt.subplots()

for result, legend in zip(result_list, legend_list):
	#y_axis = [percent_solved_n_checked(result, x) for x in x_axis]

	plt.plot(result[0], result[1], label=legend, linewidth=4.0, marker="o")

ax.set(title='Frontiers', ylabel='Beam size', xlabel='Number of candidates evaluated per problem')
ax.legend(loc='best')
savefile='plots/' + 'frontier' + '.eps'
#savefile = 'plots/time_prelim.png'
plt.savefig(savefile)