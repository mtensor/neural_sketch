#for frontier graph:
sbatch execute_gpu.sh python evaluate_robustfill.py --test_generalization --resultsfile 'rb_timeout1challenge' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_1.0_1537385508985/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11750223 
sbatch execute_gpu.sh python evaluate_robustfill.py --test_generalization --resultsfile 'rb_timeout.5challenge' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.5_1537385713113/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11750224  
sbatch execute_gpu.sh python evaluate_robustfill.py --test_generalization --resultsfile 'rb_timeout.25challenge' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.25_1537385744004/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11750225   
sbatch execute_gpu.sh python evaluate_robustfill.py --test_generalization --resultsfile 'rb_timeout.1challenge' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.10_1537401029301/robustfill_holes.p_iter_3600.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11750226  
sbatch execute_gpu.sh python evaluate_robustfill.py --test_generalization --resultsfile 'rb_rnnbase_challenge' --precomputed_data_file 'rb_challenge_tasks.p' --pretrained --pretrained_model_path './experiments/rb_long_pretrain_1537123008935/robustfill_pretrained.p'
Submitted batch job 11750227     
sbatch execute_gpu.sh python evaluate_robustfill.py --test_generalization --resultsfile 'rb_dc_base_challenge' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --dc_baseline --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11750228  

-#-#-#-#-#-#-#-#-#-#-#

sbatch execute_gpu.sh python evaluate_robustfill.py --n_samples 20 --test_generalization --resultsfile 'rb_timeout1challenge20' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_1.0_1537385508985/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11765100  
sbatch execute_gpu.sh python evaluate_robustfill.py --n_samples 20 --test_generalization --resultsfile 'rb_timeout.5challenge20' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.5_1537385713113/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11765101 
sbatch execute_gpu.sh python evaluate_robustfill.py --n_samples 20 --test_generalization --resultsfile 'rb_timeout.25challenge20' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.25_1537385744004/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11765102  
sbatch execute_gpu.sh python evaluate_robustfill.py --n_samples 20 --test_generalization --resultsfile 'rb_timeout.1challenge20' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.10_1537401029301/robustfill_holes.p_iter_3600.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11765103  
sbatch execute_gpu.sh python evaluate_robustfill.py --n_samples 20 --test_generalization --resultsfile 'rb_rnnbase_challenge20' --precomputed_data_file 'rb_challenge_tasks.p' --pretrained --pretrained_model_path './experiments/rb_long_pretrain_1537123008935/robustfill_pretrained.p'
Submitted batch job 11765104

-#-#-#-#-#-#-#-#-#-#-#

sbatch execute_gpu.sh python evaluate_robustfill.py --n_samples 50 --test_generalization --resultsfile 'rb_timeout1challenge50' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_1.0_1537385508985/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11765155
sbatch execute_gpu.sh python evaluate_robustfill.py --n_samples 50 --test_generalization --resultsfile 'rb_timeout.5challenge50' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.5_1537385713113/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11765156    
sbatch execute_gpu.sh python evaluate_robustfill.py --n_samples 50 --test_generalization --resultsfile 'rb_timeout.25challenge50' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.25_1537385744004/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11765157
sbatch execute_gpu.sh python evaluate_robustfill.py --n_samples 50 --test_generalization --resultsfile 'rb_timeout.1challenge50' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.10_1537401029301/robustfill_holes.p_iter_3600.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11765158   
sbatch execute_gpu.sh python evaluate_robustfill.py --n_samples 50 --test_generalization --resultsfile 'rb_rnnbase_challenge50' --precomputed_data_file 'rb_challenge_tasks.p' --pretrained --pretrained_model_path './experiments/rb_long_pretrain_1537123008935/robustfill_pretrained.p'
Submitted batch job 11765159 

-#-#-#-#-#-#-#-#-#-#-#


sbatch execute_any_gpu.sh python evaluate_robustfill.py --n_samples 100 --test_generalization --resultsfile 'rb_timeout1challenge100' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_1.0_1537385508985/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11765463
sbatch execute_any_gpu.sh python evaluate_robustfill.py --n_samples 100 --test_generalization --resultsfile 'rb_timeout.5challenge100' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.5_1537385713113/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11765464
sbatch execute_any_gpu.sh python evaluate_robustfill.py --n_samples 100 --test_generalization --resultsfile 'rb_timeout.25challenge100' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.25_1537385744004/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11765465  
sbatch execute_any_gpu.sh python evaluate_robustfill.py --n_samples 100 --test_generalization --resultsfile 'rb_timeout.1challenge100' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.10_1537401029301/robustfill_holes.p_iter_3600.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11765466   
sbatch execute_any_gpu.sh python evaluate_robustfill.py --n_samples 100 --test_generalization --resultsfile 'rb_rnnbase_challenge100' --precomputed_data_file 'rb_challenge_tasks.p' --pretrained --pretrained_model_path './experiments/rb_long_pretrain_1537123008935/robustfill_pretrained.p'
Submitted batch job 11765467   

-#-#-#-#-#-#-#-#-#-#-#


sbatch execute_any_gpu.sh python evaluate_robustfill.py --n_samples 5 --test_generalization --resultsfile 'rb_timeout1challenge5' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_1.0_1537385508985/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11770142
sbatch execute_any_gpu.sh python evaluate_robustfill.py --n_samples 5 --test_generalization --resultsfile 'rb_timeout.5challenge5' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.5_1537385713113/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11770143
sbatch execute_any_gpu.sh python evaluate_robustfill.py --n_samples 5 --test_generalization --resultsfile 'rb_timeout.25challenge5' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.25_1537385744004/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11770144
sbatch execute_any_gpu.sh python evaluate_robustfill.py --n_samples 5 --test_generalization --resultsfile 'rb_timeout.1challenge5' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.10_1537401029301/robustfill_holes.p_iter_3600.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11770145
sbatch execute_any_gpu.sh python evaluate_robustfill.py --n_samples 5 --test_generalization --resultsfile 'rb_rnnbase_challenge5' --precomputed_data_file 'rb_challenge_tasks.p' --pretrained --pretrained_model_path './experiments/rb_long_pretrain_1537123008935/robustfill_pretrained.p'
Submitted batch job 11770146

-#-#-#-#-#-#-#-#-#-#-#


sbatch execute_any_gpu.sh python evaluate_robustfill.py --n_samples 150 --test_generalization --resultsfile 'rb_timeout1challenge150' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_1.0_1537385508985/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11770152 
sbatch execute_any_gpu.sh python evaluate_robustfill.py --n_samples 150 --test_generalization --resultsfile 'rb_timeout.5challenge150' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.5_1537385713113/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11770148 
sbatch execute_any_gpu.sh python evaluate_robustfill.py --n_samples 150 --test_generalization --resultsfile 'rb_timeout.25challenge150' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.25_1537385744004/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11770149 
sbatch execute_any_gpu.sh python evaluate_robustfill.py --n_samples 150 --test_generalization --resultsfile 'rb_timeout.1challenge150' --precomputed_data_file 'rb_challenge_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.10_1537401029301/robustfill_holes.p_iter_3600.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11770150 
sbatch execute_any_gpu.sh python evaluate_robustfill.py --n_samples 150 --test_generalization --resultsfile 'rb_rnnbase_challenge150' --precomputed_data_file 'rb_challenge_tasks.p' --pretrained --pretrained_model_path './experiments/rb_long_pretrain_1537123008935/robustfill_pretrained.p'
Submitted batch job 11770151 


-#-#-#-#-#-#-#-#-#-#-#
DEEPCODER:


sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_0.1' --precomputed_data_file 'data/prelim_test_data.p' --n_test 500 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.10_1537400900116/deepcoder_holes.p'
Submitted batch job 11751302 
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_0.25' --precomputed_data_file 'data/prelim_test_data.p' --n_test 500 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.25_1537327473486/deepcoder_holes.p' 
Submitted batch job 11751303 
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_0.5' --precomputed_data_file 'data/prelim_test_data.p' --n_test 500 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.5_1537326548865/deepcoder_holes.p' 
Submitted batch job 11751304  
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_1' --precomputed_data_file 'data/prelim_test_data.p' --n_test 500 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_1.0_1537326582936/deepcoder_holes.p'
Submitted batch job 11751305 
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_rl' --precomputed_data_file 'data/prelim_test_data.p' --n_test 500 --dcModel --mdl 14 --model_path 'experiments/deepcoder_rl_exp86_0.5_start_holes_0.0001_var_red_1536820812072/deepcoder_holes.p'
Submitted batch job 11750364 
#rnn baseline
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_rnn' --precomputed_data_file 'data/prelim_test_data.p' --n_test 500 --dcModel --mdl 14 --model_path 'experiments/deepcoder_long_rnn_base_1537308558965/deepcoder_rnn_base.p' 
Submitted batch job 11750365 
#dc baseline 
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_dcbase' --dc_baseline --precomputed_data_file 'data/prelim_test_data.p' --n_test 500 --dcModel --mdl 14
Submitted batch job 11750366

-#-#-#-#-#-#-#-#-#-#-#

sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_0.1_30' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.10_1537400900116/deepcoder_holes.p'

sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_0.25_30' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.25_1537327473486/deepcoder_holes.p' 

sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_0.5_30' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.5_1537326548865/deepcoder_holes.p' 

sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_1_30' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_1.0_1537326582936/deepcoder_holes.p'

sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_rl_30' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_rl_exp86_0.5_start_holes_0.0001_var_red_1536820812072/deepcoder_holes.p'

#rnn baseline
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_rnn_30' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_long_rnn_base_1537308558965/deepcoder_rnn_base.p' 

#dc baseline 
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_dcbase_30' --dc_baseline --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14


-#-#-#-#-#-#-#-#-#-#-#

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 50 --resultsfile 'dc_timeout_0.1_50' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.10_1537400900116/deepcoder_holes.p'

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 50 --resultsfile 'dc_timeout_0.25_50' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.25_1537327473486/deepcoder_holes.p' 

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 50 --resultsfile 'dc_timeout_0.5_50' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.5_1537326548865/deepcoder_holes.p' 

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 50 --resultsfile 'dc_timeout_1_50' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_1.0_1537326582936/deepcoder_holes.p'

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 50 --resultsfile 'dc_rl_50' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_rl_exp86_0.5_start_holes_0.0001_var_red_1536820812072/deepcoder_holes.p'

#rnn baseline
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 50 --resultsfile 'dc_rnn_50' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_long_rnn_base_1537308558965/deepcoder_rnn_base.p' 


-#-#-#-#-#-#-#-#-#-#-#

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 100 --resultsfile 'dc_timeout_0.1_100' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.10_1537400900116/deepcoder_holes.p'

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 100 --resultsfile 'dc_timeout_0.25_100' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.25_1537327473486/deepcoder_holes.p' 

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 100 --resultsfile 'dc_timeout_0.5_100' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.5_1537326548865/deepcoder_holes.p' 

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 100 --resultsfile 'dc_timeout_1_100' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_1.0_1537326582936/deepcoder_holes.p'

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 100 --resultsfile 'dc_rl_100' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_rl_exp86_0.5_start_holes_0.0001_var_red_1536820812072/deepcoder_holes.p'

#rnn baseline
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 100 --resultsfile 'dc_rnn_100' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_long_rnn_base_1537308558965/deepcoder_rnn_base.p' 

-#-#-#-#-#-#-#-#-#-#-#

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 150 --resultsfile 'dc_timeout_0.1_150' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.10_1537400900116/deepcoder_holes.p'

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 150 --resultsfile 'dc_timeout_0.25_150' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.25_1537327473486/deepcoder_holes.p' 

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 150 --resultsfile 'dc_timeout_0.5_150' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.5_1537326548865/deepcoder_holes.p' 

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 150 --resultsfile 'dc_timeout_1_150' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_1.0_1537326582936/deepcoder_holes.p'

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 150 --resultsfile 'dc_rl_150' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_rl_exp86_0.5_start_holes_0.0001_var_red_1536820812072/deepcoder_holes.p'

#rnn baseline
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 150 --resultsfile 'dc_rnn_150' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_long_rnn_base_1537308558965/deepcoder_rnn_base.p' 


-#-#-#-#-#-#-#-#-#-#-#

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 5 --resultsfile 'dc_timeout_0.1_5' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.10_1537400900116/deepcoder_holes.p'

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 5 --resultsfile 'dc_timeout_0.25_5' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.25_1537327473486/deepcoder_holes.p' 

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 5 --resultsfile 'dc_timeout_0.5_5' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.5_1537326548865/deepcoder_holes.p' 

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 5 --resultsfile 'dc_timeout_1_5' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_1.0_1537326582936/deepcoder_holes.p'

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 5 --resultsfile 'dc_rl_5' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_rl_exp86_0.5_start_holes_0.0001_var_red_1536820812072/deepcoder_holes.p'

#rnn baseline
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 5 --resultsfile 'dc_rnn_5' --precomputed_data_file 'data/prelim_test_data.p' --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_long_rnn_base_1537308558965/deepcoder_rnn_base.p' 



