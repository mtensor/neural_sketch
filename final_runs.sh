
TESTING:
-[ ] long tests on testing data,
-[ ] use the "real" sygis data to make the correctness graphs Armando wants
-[ ] the samples vs enum budget graph

[ ] evaluate rnn baseline (vary sample num)
[ ] evalute dc baseline 
[ ] evaluate sketch model 4x (vary sample num)

--test_generalization flag can be used for sygis data testing, still need to write plotter to plot g_hit/hit as function of enum budget




TODO: 
-[X] decision on which types of models still to train
-[ ] evaluation: write all of the scripts up for eval and for plotting 
	-[ ] for ease, make eval take a file path as input, so i dont have to look things up
	-[ ] long general runs (DC and RB)
	-[ ] real data graphs (RB)
	-[ ] frontier graphs (DC + RB)?





need to run:
-[ ] a good T4 evaluation - waiting on this, 100k may be all we get on this one

-[ ] 100K T3

##########t3 100k run###########
NEED TO RUN!!! (if I can)
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_0.25_40k' --beam --n_samples 50 --shuffled --max_to_check 40000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_0.25_1537327473486/deepcoder_holes.p' 
Submitted batch job 11783626
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_0.5_40k' --beam --n_samples 50 --shuffled --max_to_check 40000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_0.5_1537326548865/deepcoder_holes.p' 
Submitted batch job 11783627
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_1_40k' --beam --n_samples 50 --shuffled --max_to_check 40000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_1.0_1537326582936/deepcoder_holes.p'
Submitted batch job 11783628
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_dcbase_40k' --beam --n_samples 50 --shuffled --max_to_check  40000 --dc_baseline --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 18
Submitted batch job 11783629
-#-#-#-#-#-#-#-#-#-#-#
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_0.25_40k_beam100' --beam --n_samples 100 --shuffled --max_to_check 40000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_0.25_1537327473486/deepcoder_holes.p' 
Submitted batch job 11788402
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_0.5_40k_beam100' --beam --n_samples 100 --shuffled --max_to_check 40000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_0.5_1537326548865/deepcoder_holes.p' 
Submitted batch job 11788403
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_1_40k_beam100' --beam --n_samples 100 --shuffled --max_to_check 40000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_1.0_1537326582936/deepcoder_holes.p'
Submitted batch job 11788404
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_rnn_beam_50_long_beam100' --shuffled --beam --n_samples 100 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 14 --model_path 'experiments/deepcoder_long_rnn_base_1537308558965/deepcoder_rnn_base.p' 
Submitted batch job 11788405
-#-#-#-#-#-#-#-#-#-#-#
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_0.25_40k_beam20' --beam --n_samples 20 --shuffled --max_to_check 40000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_0.25_1537327473486/deepcoder_holes.p' 
Submitted batch job 11788411
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_0.5_40k_beam20' --beam --n_samples 20 --shuffled --max_to_check 40000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_0.5_1537326548865/deepcoder_holes.p' 
Submitted batch job 11788412
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_1_40k_beam20' --beam --n_samples 20 --shuffled --max_to_check 40000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_1.0_1537326582936/deepcoder_holes.p'
Submitted batch job 11788413
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_rnn_beam_50_long_beam20' --shuffled --beam --n_samples 20 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 14 --model_path 'experiments/deepcoder_long_rnn_base_1537308558965/deepcoder_rnn_base.p' 
Submitted batch job 11788414
-#-#-#-#-#-#-#-#-#-#-#
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_0.25_40k_beam5' --beam --n_samples 5 --shuffled --max_to_check 40000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_0.25_1537327473486/deepcoder_holes.p' 
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_0.5_40k_beam5' --beam --n_samples 5 --shuffled --max_to_check 40000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_0.5_1537326548865/deepcoder_holes.p' 
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_1_40k_beam5' --beam --n_samples 5 --shuffled --max_to_check 40000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_1.0_1537326582936/deepcoder_holes.p'
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_rnn_beam_50_long_beam5' --shuffled --beam --n_samples 5 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 14 --model_path 'experiments/deepcoder_long_rnn_base_1537308558965/deepcoder_rnn_base.p' 



-[ ] T3 rnn with beam x2 (one for short eval and one for long eval)

sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_rnn_beam_50_long' --shuffled --beam --n_samples 50 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 14 --model_path 'experiments/deepcoder_long_rnn_base_1537308558965/deepcoder_rnn_base.p' 
Submitted batch job 11783625
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_rnn_beam_50' --beam --n_samples 50 --precomputed_data_file 'data/prelim_test_data.p' --n_test 500 --dcModel --mdl 14 --model_path 'experiments/deepcoder_long_rnn_base_1537308558965/deepcoder_rnn_base.p' 

sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_rnn_beam_50_long_non_shuffled' --beam --n_samples 50 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 14 --model_path 'experiments/deepcoder_long_rnn_base_1537308558965/deepcoder_rnn_base.p' 
Submitted batch job 11783624


-------------
anaconda-project run python make_final_plots.py --filename 'dc_T3_beam50' --file_list 'results/dc_timeout_0.25_40k.p' 'results/dc_timeout_0.5_40k.p' 'results/dc_timeout_1_40k.p' 'results/dc_dcbase_40k.p' 'results/dc_rnn_beam_50_long.p' --legend_list "Flexible_neural_sketch,_decay=0.25_(ours)" "Flexible_neural_sketch,_decay=0.5_(ours)" "Flexible_neural_sketch,_decay=1.0_(ours)" "Deepcoder_baseline" "RNN_baseline"




-------------

-[ ] string rnn base beam search
	-[X] 50
	-[X] 100
	-[X] 150 -- this is sooo slow .... 

-[ ] string sketch model --- probably 50 and 100



-[ ] the big tradeoff plot
	What do i need for this?????
	- do I need to redo everything with a beam search??
	- can I just do different amounts of sampling??

	- could just do for string tasks, could get away with running only 20 jobs to do this [5, 50, 100, 150]*5, yeah, but they would take a very long time ... 




-#-#-#-#-#-#-#-#-#-#-#

sbatch execute_gpu.sh python evaluate_robustfill.py --beam --n_samples 50 --test_generalization --resultsfile 'rb_timeout1challenge50beam' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_1.0_1537385508985/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
sbatch execute_gpu.sh python evaluate_robustfill.py --beam --n_samples 50 --test_generalization --resultsfile 'rb_timeout.5challenge50beam' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.5_1537385713113/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
sbatch execute_gpu.sh python evaluate_robustfill.py --beam --n_samples 50 --test_generalization --resultsfile 'rb_timeout.25challenge50beam' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.25_1537385744004/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
sbatch execute_gpu.sh python evaluate_robustfill.py --beam --n_samples 50 --test_generalization --resultsfile 'rb_timeout.1challenge50beam' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.10_1537401029301/robustfill_holes.p_iter_3600.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 

-#-#-#-#-#-#-#-#-#-#-#

sbatch execute_any_gpu.sh python evaluate_robustfill.py --beam --n_samples 100 --test_generalization --resultsfile 'rb_timeout1challenge100beam' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_1.0_1537385508985/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11782239
sbatch execute_any_gpu.sh python evaluate_robustfill.py --beam --n_samples 100 --test_generalization --resultsfile 'rb_timeout.5challenge100beam' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.5_1537385713113/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11782241
sbatch execute_any_gpu.sh python evaluate_robustfill.py --beam --n_samples 100 --test_generalization --resultsfile 'rb_timeout.25challenge100beam' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.25_1537385744004/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11782242
sbatch execute_any_gpu.sh python evaluate_robustfill.py --beam --n_samples 100 --test_generalization --resultsfile 'rb_timeout.1challenge100beam' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.10_1537401029301/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11782244

-#-#-#-#-#-#-#-#-#-#-#

sbatch execute_any_gpu.sh python evaluate_robustfill.py --beam --n_samples 5 --test_generalization --resultsfile 'rb_timeout1challenge5beam' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_1.0_1537385508985/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
sbatch execute_any_gpu.sh python evaluate_robustfill.py --beam --n_samples 5 --test_generalization --resultsfile 'rb_timeout.5challenge5beam' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.5_1537385713113/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
sbatch execute_any_gpu.sh python evaluate_robustfill.py --beam --n_samples 5 --test_generalization --resultsfile 'rb_timeout.25challenge5beam' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.25_1537385744004/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
sbatch execute_any_gpu.sh python evaluate_robustfill.py --beam --n_samples 5 --test_generalization --resultsfile 'rb_timeout.1challenge5beam' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.10_1537401029301/robustfill_holes.p_iter_3600.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
sbatch execute_any_gpu.sh python evaluate_robustfill.py --beam --n_samples 5 --test_generalization --resultsfile 'rb_rnnbase_challenge5beam' --precomputed_data_file 'rb_challenge_tasks.p' --pretrained --pretrained_model_path './experiments/rb_long_pretrain_1537123008935/robustfill_pretrained.p'

-#-#-#-#-#-#-#-#-#-#-#

sbatch execute_any_gpu.sh python evaluate_robustfill.py --beam --n_samples 150 --test_generalization --resultsfile 'rb_timeout1challenge150beam' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_1.0_1537385508985/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
sbatch execute_any_gpu.sh python evaluate_robustfill.py --beam --n_samples 150 --test_generalization --resultsfile 'rb_timeout.5challenge150beam' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.5_1537385713113/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
sbatch execute_any_gpu.sh python evaluate_robustfill.py --beam --n_samples 150 --test_generalization --resultsfile 'rb_timeout.25challenge150beam' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.25_1537385744004/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
sbatch execute_any_gpu.sh python evaluate_robustfill.py --beam --n_samples 150 --test_generalization --resultsfile 'rb_timeout.1challenge150beam' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.10_1537401029301/robustfill_holes.p_iter_3600.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
