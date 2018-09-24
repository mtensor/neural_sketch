#make_final_plots
from manipulate_results import plot_result_list
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--legend_list', nargs='*')
parser.add_argument('--file_list', nargs='*')
parser.add_argument('--filename', type=str)
parser.add_argument('--robustfill', action='store_true')
parser.add_argument('--generalization', action='store_true')
args = parser.parse_args()

if __name__=='__main__':
	args.legend_list = [l.replace('_', ' ') for l in args.legend_list]
	print(args.legend_list)
	plot_result_list(args.file_list, args.legend_list, args.filename, robustfill=args.robustfill, plot_time=not args.generalization, generalization=args.generalization)