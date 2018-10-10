#pypy_runs.sh




#test pypy
python evaluate_deepcoder.py --resultsfile 'dc_timeout_0.25_40k_T34pypy' --beam --n_samples 100 --shuffled --max_to_check 40000 --precomputed_data_file 'data/T4_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_0.25_1537327473486/deepcoder_holes.p' 

python evaluate_deepcoder.py --resultsfile 'dc_timeout_0.25_40k_beam100pypy' --beam --n_samples 50 --shuffled --max_to_check 40000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_0.25_1537327473486/deepcoder_holes.p' 

python evaluate_deepcoder.py --resultsfile 'dc_dcbase_400k_T34pypy' --beam --n_samples 100 --shuffled --max_to_check  200000 --dc_baseline --precomputed_data_file 'data/T4_test_data.p' --n_test 100 --dcModel --mdl 18





#need to rerun:

#LIST TASKS:
#ours, .25, t33
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T33_0.25_800k_pypy' --beam --n_samples 100 --shuffled --max_to_check 800000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_0.25_1537327473486/deepcoder_holes.p' 
Submitted batch job 11966289  
#dc baseline, t33
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T33_dcbase_800k_pypy' --beam --n_samples 100 --shuffled --max_to_check  800000 --dc_baseline --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 18
Submitted batch job 11966325

#ours, .25, t34
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T34_0.25_800k_pypy' --beam --n_samples 100 --max_to_check 800000 --precomputed_data_file 'data/T4_test_data.p' --n_test 100 --dcModel --mdl 17 --model_path 'experiments/deepcoder_timeout_0.25_1537327473486/deepcoder_holes.p' 
Submitted batch job 11966846  
#dc baseline, t34
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T34_dcbase_800k_pypy' --beam --n_samples 100 --max_to_check  800000 --dc_baseline --precomputed_data_file 'data/T4_test_data.p' --n_test 100 --dcModel --mdl 18
Submitted batch job 11966847

#ours, .25, t45
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T45_0.25_800k_pypy' --beam --n_samples 100 --n_test 96 --dcModel --mdl 17 --max_to_check 800000 --model_path 'experiments/deepcoder_T4_timeout_0.25_1537402389185/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11966848 -- died because of memory issues, rerun with more memory
Submitted batch job 11971095
#dc baseline, t45
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T45_dcbase_800k_pypy' --beam --n_test 96 --dcModel --mdl 18 --max_to_check 800000 --dc_baseline --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11966849






anaconda-project run python make_final_plots.py --max_budget 800000 --title "List Processing: length 4 test programs" --notime --filename 'dc_T34_pypy' --file_list 'results/dc_T34_0.25_800k_pypy.p' 'results/dc_T34_dcbase_800k_pypy.p' 'results/dc_rnn_beam_T34_2.p' --legend_list "SketchAdapt_(ours)" "Synthesizer_only_(Deepcoder)" "Generator_only_(RobustFill)"
-- submitted

python make_final_plots.py --max_budget 800000 --title "List Processing: length 3 test programs" --notime --filename 'dc_T33_pypy' --file_list 'results/dc_T33_0.25_800k_pypy.p' 'results/dc_T33_dcbase_800k_pypy.p' 'results/dc_rnn_beam_50_long_beam100.p' --legend_list "SketchAdapt_(ours)" "Synthesizer_only_(Deepcoder)" "Generator_only_(RobustFill)"
-- submitted

python make_final_plots.py --max_budget 800000 --title "List Processing: length 5 test programs (trained on length 4)" --notime --filename 'dc_T45_pypy' --file_list 'results/dc_T45_0.25_800k_pypy.p' 'results/dc_T45_dcbase_800k_pypy.p' 'results/dc_T4_rnn_base_beam_100k.p' --legend_list "SketchAdapt_(ours)" "Synthesizer_only_(Deepcoder)" "Generator_only_(RobustFill)"
-- had memory issue!!!


python make_final_plots.py --max_budget 40000 --title "String Editing: SyGuS dataset" --robustfill --notime --filename 'rb_challenge_pypy_log' --file_list 'rb_results/rb_0.25challenge_100_pypy.p' 'rb_results/rb_0.25challenge_50_pypy.p' 'rb_results/rb_rnnbase_challenge_beam_100.p' 'rb_results/rb_rnnbase_challenge_beam_50.p' 'rb_results/rb_dc_base_challenge_pypy.p' --legend_list "SketchAdapt,_beam_100_(ours)" "SketchAdapt,_beam_50_(ours)" "Generator_only,_beam_100_(RobustFill)" "Generator_only,_beam_50_(RobustFill)" "Synthesizer_only_(Deepcoder)"

python make_final_plots.py --max_budget 40000 --title "String Editing" --robustfill --notime --filename 'rb_test_pypy' --file_list 'rb_results/rb_0.25test_100_pypy.p' 'rb_results/rb_0.25test_50_pypy.p' 'rb_results/rb_rnnbase_test_beam100.p' 'rb_results/rb_rnnbase_test_beam50.p' 'rb_results/rb_dc_base_test_pypy.p' --legend_list "SketchAdapt,_beam_100_(ours)" "SketchAdapt,_beam_50_(ours)" "Generator_only,_beam_100_(RobustFill)" "Generator_only,_beam_50_(RobustFill)" "Synthesizer_only_(Deepcoder)"


python make_final_plots.py --max_budget 40000 --robustfill --notime --filename 'rb_test_pypy_no100_log' --file_list  'rb_results/rb_0.25test_50_pypy.p' 'rb_results/rb_rnnbase_test_beam100.p' 'rb_results/rb_rnnbase_test_beam50.p' 'rb_results/rb_dc_base_test_pypy.p' --legend_list "SketchAdapt,_beam_50_(ours)" "RNN_baseline,_beam_100" "RNN_baseline,_beam_50" "Deepcoder_baseline"



#STRING TESTS:

#challenge:
#ours, .25 beam 100
sbatch execute_gpu.sh python evaluate_robustfill.py --max_to_check 40000 --mdl 17 --beam --n_samples 100 --test_generalization --resultsfile 'rb_0.25challenge_100_pypy' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.25_1537385744004/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11967124
#ours, .25 beam 50
sbatch execute_gpu.sh python evaluate_robustfill.py --max_to_check 40000 --mdl 17 --beam --n_samples 50 --test_generalization --resultsfile 'rb_0.25challenge_50_pypy' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.25_1537385744004/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11967125
#dc baseline
sbatch execute_gpu.sh python evaluate_robustfill.py --max_to_check 40000 --mdl 17 --test_generalization --resultsfile 'rb_dc_base_challenge_pypy' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --dc_baseline --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11967126

#kevin's:
#ours, .25 beam 100
sbatch execute_gpu.sh python evaluate_robustfill.py --max_to_check 40000 --mdl 17 --beam --n_samples 100 --resultsfile 'rb_0.25test_100_pypy' --precomputed_data_file 'rb_test_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.25_1537385744004/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11967127

#ours, .25 beam 50
sbatch execute_gpu.sh python evaluate_robustfill.py --max_to_check 40000 --mdl 17 --beam --n_samples 50 --resultsfile 'rb_0.25test_50_pypy' --precomputed_data_file 'rb_test_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.25_1537385744004/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11967128
#dc baseline
sbatch execute_gpu.sh python evaluate_robustfill.py --max_to_check 40000 --mdl 17 --test_generalization --resultsfile 'rb_dc_base_test_pypy' --precomputed_data_file 'rb_test_tasks.p' --dcModel --dc_baseline --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11967129



#########

sbatch execute_gpu.sh python evaluate_robustfill.py --max_to_check 400000 --mdl 19 --test_generalization --resultsfile 'rb_dc_base_challenge_pypy_long' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --dc_baseline --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
11977767
sbatch execute_gpu.sh python evaluate_robustfill.py --max_to_check 400000 --mdl 19 --test_generalization --resultsfile 'rb_dc_base_test_pypy_long' --precomputed_data_file 'rb_test_tasks.p' --dcModel --dc_baseline --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
11977768

sbatch execute_gpu.sh python evaluate_robustfill.py --max_to_check 1000000 --mdl 20 --test_generalization --resultsfile 'rb_dc_base_challenge_pypy_xlong' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --dc_baseline --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11977773
sbatch execute_gpu.sh python evaluate_robustfill.py --max_to_check 1000000 --mdl 20 --test_generalization --resultsfile 'rb_dc_base_test_pypy_xlong' --precomputed_data_file 'rb_test_tasks.p' --dcModel --dc_baseline --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11977774

########



#SUPERLONG DC:

sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T33_0.25_2M_pypy' --beam --n_samples 100 --shuffled --max_to_check 2000000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 17 --model_path 'experiments/deepcoder_timeout_0.25_1537327473486/deepcoder_holes.p' 
 Submitted batch job 11967134
#dc baseline, t33
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T33_dcbase_2M_pypy' --beam --n_samples 100 --shuffled --max_to_check  2000000 --dc_baseline --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 18
Submitted batch job 11967135

#ours, .25, t34
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T34_0.25_2M_pypy' --beam --n_samples 100 --max_to_check 2000000 --precomputed_data_file 'data/T4_test_data.p' --n_test 100 --dcModel --mdl 17 --model_path 'experiments/deepcoder_timeout_0.25_1537327473486/deepcoder_holes.p' 
Submitted batch job 11967136 -- memory issue 
#dc baseline, t34
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T34_dcbase_2M_pypy' --beam --n_samples 100 --max_to_check  2000000 --dc_baseline --precomputed_data_file 'data/T4_test_data.p' --n_test 100 --dcModel --mdl 18
Submitted batch job 11967137

#ours, .25, t45
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T45_0.25_2M_pypy' --beam --n_samples 100 --n_test 96 --dcModel --mdl 17 --max_to_check 2000000 --model_path 'experiments/deepcoder_T4_timeout_0.25_1537402389185/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11967138 -- memory issue
#dc baseline, t45
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T45_dcbase_2M_pypy' --beam --n_test 96 --dcModel --mdl 18 --max_to_check 2000000 --dc_baseline --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11967139



python make_final_plots.py --title "Length 4 test programs" --max_budget 2000000 --notime --filename 'dc_T34_2M_pypy' --file_list 'results/dc_T34_0.25_2M_pypy.p' 'results/dc_T34_dcbase_2M_pypy.p' 'results/dc_rnn_beam_T34_2.p' --legend_list "SketchAdapt_(ours)" "Deepcoder_baseline" "RNN_baseline"

python make_final_plots.py --title "Trained on length 4 programs, tested on length 5 programs" --max_budget 2000000 --notime --filename 'dc_T45_2M_pypy' --file_list 'results/dc_T45_0.25_2M_pypy.p' 'results/dc_T45_dcbase_2M_pypy.p' 'results/dc_T4_rnn_base_beam_100k.p' --legend_list "SketchAdapt_(ours)" "Deepcoder_baseline" "RNN_baseline"


TODO:
deal with superlong runs
deal with mem issue runs


from manipulate_results import rload

'results/dc_T45_0.25_800k_pypy.p'
'rb_results/rb_0.25challenge_100_pypy.p'
'rb_results/rb_0.25test_100_pypy.p'
'results/dc_T33_0.25_800k_pypy.p'
'results/dc_T34_0.25_800k_pypy.p'

#filter results by hit tasks
hits = [(task.IO, result) for task, result_list in results.items() for result in result_list if result.hit]
