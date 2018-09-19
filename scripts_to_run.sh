# scripts to run






#DEEPCODER RUNS TO DO:

- grammar to poke holes (hole training time) 


- pretrain on full 512 IO:
name=deepcoder_pretrained_512 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --pretrain --max_epochs 0 --max_pretrain_epochs 10 --Vrange 512 --new --save_pretrained_model_path "./deepcoder_pretrained_512.p"
'experiments/deepcoder_pretrained_512_1536675470181'

name=deepcoder_pretrained_V128_10_epochs g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --pretrain --max_epochs 0 --max_pretrain_epochs 10 --Vrange 128 --save_pretrained_model_path "./deepcoder_pretrained.p"
'experiments/deepcoder_pretrained_V128_10_epochs_1536681569098' -- is this bad because of overfitting? - take a look at it

- train dc model on 512:
name=deepcoder_dcModel_512 g-run sbatch execute_gpu.sh python deepcoder_train_dc_model.py --save_model_path './dc_model_512.p' --new
'experiments/deepcoder_dcModel_512_1536676494070'


- train and test on 3 w holes 
name=deepcoder_T3_V128_temp0.5 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs todo --Vrange 128 --load_pretrained_model_path "./deepcoder_pretrained.p" --use_dc_grammar grammar_path --inv_temp 0.5
name=deepcoder_T3_V128_temp1.0 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs todo --Vrange 128 --load_pretrained_model_path "./deepcoder_pretrained.p" --use_dc_grammar grammar_path --inv_temp 1.0

name=deepcoder_T3_V512_temp0.5 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs todo --Vrange 512 --load_pretrained_model_path "./deepcoder_pretrained_512.p" --use_dc_grammar grammar_path --inv_temp 0.5
name=deepcoder_T3_V512_temp1.0 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs todo --Vrange 512 --load_pretrained_model_path "./deepcoder_pretrained_512.p" --use_dc_grammar grammar_path --inv_temp 1.0

- train on 4, test on 5 (both pretrain and holetraining time)
	- #both 128 and 512 or just 512??

name=deepcoder_T4_V128_temp0.5 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs todo --Vrange 128 --load_pretrained_model_path "./deepcoder_pretrained.p" --datas [TODO] --use_dc_grammar grammar_path --inv_temp 0.5
name=deepcoder_T4_V128_temp1.0 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs todo --Vrange 128 --load_pretrained_model_path "./deepcoder_pretrained.p" --datas [TODO] --use_dc_grammar grammar_path --inv_temp 1.0

name=deepcoder_T4_V512_temp0.5 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs todo --Vrange 512 --load_pretrained_model_path "./deepcoder_pretrained_512.p" --datas [TODO] --use_dc_grammar grammar_path --inv_temp 0.5
name=deepcoder_T4_V512_temp1.0 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs todo --Vrange 512 --load_pretrained_model_path "./deepcoder_pretrained_512.p" --datas [TODO] --use_dc_grammar grammar_path --inv_temp 1.0

- multiple n_samples (eval time)
	- need to eval all of the above
	- plus RNN baseline and DC baseline

# eval
name=deepcoder_eval_for_each sbatch gpu_execute.sh python deepcoder_evaluate.py --model_path 'deepcoder_model_holes.p' --test_data_path 'test_data'
# need for T4 and T3
# need for each temp
# need for 128 vs 512

##################### Regex:
- write regex_train_dc_model
- write/check main_supervised_regex.py
# train dc model
name=regex_train_dc_model g-run sbatch gpu_execute.sh python regex_train_dc_model.py --save_model_path './regex_dc_model.p' --new  # TODO

#pretrain regex model
name=regex_pretrained g-run sbatch execute_gpu.sh python main_supervised_regex.py --pretrain --max_epochs 0 --max_pretrain_iterations todo --new --save_pretrained_model_path "./regex_pretrained.p"

# train actual model
name=regex_train_supervised_temp0.5 g-run sbatch gpu_execute.sh python main_supervised_regex.py --max_iterations todo  --load_pretrained_model_path "./regex_pretrained.p" --use_dc_grammar grammar_path --inv_temp 0.5
name=regex_train_supervised_temp1.0 g-run sbatch gpu_execute.sh python main_supervised_regex.py --max_iterations todo  --load_pretrained_model_path "./regex_pretrained.p" --use_dc_grammar grammar_path --inv_temp 1.0
name=regex_train_supervised_temp0.75 g-run sbatch gpu_execute.sh python main_supervised_regex.py --max_iterations todo  --load_pretrained_model_path "./regex_pretrained.p" --use_dc_grammar grammar_path --inv_temp 0.75
name=regex_train_supervised_temp0.25 g-run sbatch gpu_execute.sh python main_supervised_regex.py --max_iterations todo  --load_pretrained_model_path "./regex_pretrained.p" --use_dc_grammar grammar_path --inv_temp 0.25

# regex evaluation for each
- write regex_evaluate
name=regex_eval_for_each sbatch gpu_execute.sh python regex_evaluate.py --model_path 'regex_model_holes.p' --test_data_path 'data'

name=regex_eval_for_each sbatch gpu_execute.sh python regex_evaluate.py --model_path 'regex_model_holes.p' --test_data_path 'data' --dc_baseline

name=regex_eval_for_each sbatch gpu_execute.sh python regex_evaluate.py --model_path 'regex_model_pretrained.p' --test_data_path 'data' --rnn_baseline

- need test data
anaconda_project run python manipulate_results.py $regex_results_file $regex_base_results_file # or something like that 



python main_supervised_deepcoder.py --max_epochs 10 --use_rl --variance_reduction --rl_no_syntax --nosave  --load_pretrained_model_path './experiments/deepcoder_temp_0.5_1536186291153/deepcoder_holes.p' --inv_temp 0.5 --train_data 'data/DeepCoder_data/T3_A2_V512_L10_train_perm.txt'
python main_supervised_deepcoder.py --max_epochs 10 --use_rl --variance_reduction --rl_no_syntax --nosave  --load_pretrained_model_path './experiments/deepcoder_temp_1.0_1536186516665/deepcoder_holes.p'
python main_supervised_deepcoder.py --max_epochs 10 --use_rl --rl_no_syntax --nosave  --load_pretrained_model_path './experiments/deepcoder_temp_1.0_1536186516665/deepcoder_holes.p'
python main_supervised_deepcoder.py --max_epochs 10 --use_rl --rl_no_syntax --nosave  --load_pretrained_model_path './experiments/deepcoder_temp_1.0_1536186516665/deepcoder_holes.p' --variance_reduction --train_data 'data/DeepCoder_data/T3_A2_V512_L10_train_perm.txt'

name=deepcoder_rl_0.5_start_holes_0.0001 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs 10 --use_rl --rl_no_syntax  --load_pretrained_model_path '../deepcoder_temp_0.5_1536186291153/deepcoder_holes.p' --inv_temp 0.5 --train_data 'data/DeepCoder_data/T3_A2_V512_L10_train_perm.txt' --rl_lr 0.0001
name=deepcoder_rl_0.5_start_holes_0.0001_var_red g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs 10 --use_rl --rl_no_syntax  --load_pretrained_model_path '../deepcoder_temp_0.5_1536186291153/deepcoder_holes.p' --inv_temp 0.5 --train_data 'data/DeepCoder_data/T3_A2_V512_L10_train_perm.txt' --rl_lr 0.0001 --variance_reduction
#actually track importance weighted objective#

python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_rl_0.5_start_holes_0.0001_1536727388636/deepcoder_holes.p'
'results/prelim_results_wdcModel__test50_1536770602.p'
python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_rl_0.5_start_holes_0.0001_var_red_1536727726734/deepcoder_holes.p'



#compare to:
python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_temp_0.5_1536186291153/deepcoder_holes.p'   
'results/prelim_results_wdcModel__test50_1536770193.p'

python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --dc_baseline                                      
'results/prelim_results_dc_baseline__test50_1536770416.p'

python manipulate_results.py 'results/prelim_results_dc_baseline__test50_1536770416.p' 'results/prelim_results_wdcModel__test50_1536770193.p'


#TRYING LINEAR REWARD:
name=deepcoder_rl_linear8_0.5_start_holes_0.0001 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs 10 --use_rl --rl_no_syntax  --load_pretrained_model_path '../deepcoder_temp_0.5_1536186291153/deepcoder_holes.p' --inv_temp 0.5 --train_data 'data/DeepCoder_data/T3_A2_V512_L10_train_perm.txt' --rl_lr 0.0001 --reward_fn linear --sample_fn linear --r_max 8
'experiments/deepcoder_rl_linear8_0.5_start_holes_0.0001_1536783027682'
python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_rl_linear8_0.5_start_holes_0.0001_1536783027682/deepcoder_holes.p'
'results/prelim_results_wdcModel__test50_1536822003.p' #50%


name=deepcoder_rl_linear8_0.5_start_holes_0.0001_var_red g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs 10 --use_rl --rl_no_syntax  --load_pretrained_model_path '../deepcoder_temp_0.5_1536186291153/deepcoder_holes.p' --inv_temp 0.5 --train_data 'data/DeepCoder_data/T3_A2_V512_L10_train_perm.txt' --rl_lr 0.0001 --variance_reduction --reward_fn linear --sample_fn linear --r_max 8
'experiments/deepcoder_rl_linear8_0.5_start_holes_0.0001_var_red_1536783074744'
python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_rl_linear8_0.5_start_holes_0.0001_var_red_1536783074744/deepcoder_holes.p'
'results/prelim_results_wdcModel__test50_1536805011.p'
'results/prelim_results_wdcModel__test50_1536819663.p'
#running

#TRYING EXPONENTIAL REWARD:
name=deepcoder_rl_exp84_0.5_start_holes_0.0001 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs 10 --use_rl --rl_no_syntax  --load_pretrained_model_path '../deepcoder_temp_0.5_1536186291153/deepcoder_holes.p' --inv_temp 0.5 --train_data 'data/DeepCoder_data/T3_A2_V512_L10_train_perm.txt' --rl_lr 0.0001 --reward_fn exp --sample_fn exp --r_max 8 --num_half_lifes 4
'experiments/deepcoder_rl_exp84_0.5_start_holes_0.0001_1536787285025'
python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_rl_exp84_0.5_start_holes_0.0001_var_red_1536787304031/deepcoder_holes.p'

#running

name=deepcoder_rl_exp84_0.5_start_holes_0.0001_var_red g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs 10 --use_rl --rl_no_syntax  --load_pretrained_model_path '../deepcoder_temp_0.5_1536186291153/deepcoder_holes.p' --inv_temp 0.5 --train_data 'data/DeepCoder_data/T3_A2_V512_L10_train_perm.txt' --rl_lr 0.0001 --variance_reduction --reward_fn exp --sample_fn exp --r_max 8 --num_half_lifes 4
'experiments/deepcoder_rl_exp84_0.5_start_holes_0.0001_var_red_1536787304031'
python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_rl_exp84_0.5_start_holes_0.0001_var_red_1536787304031/deepcoder_holes.p'
'results/prelim_results_wdcModel__test50_1536804657.p'
'results/prelim_results_wdcModel__test50_1536804719.p'

name=deepcoder_rl_exp86_0.5_start_holes_0.0001 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs 10 --use_rl --rl_no_syntax  --load_pretrained_model_path '../deepcoder_temp_0.5_1536186291153/deepcoder_holes.p' --inv_temp 0.5 --train_data 'data/DeepCoder_data/T3_A2_V512_L10_train_perm.txt' --rl_lr 0.0001 --reward_fn exp --sample_fn exp --r_max 8 --num_half_lifes 6
'experiments/deepcoder_rl_exp86_0.5_start_holes_0.0001_1536787506496'
python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_rl_exp86_0.5_start_holes_0.0001_1536787506496/deepcoder_holes.p'
'results/prelim_results_wdcModel__test50_1536803560.p'
'results/prelim_results_wdcModel__test50_1536818453.p'


sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --precomputed_data_file data/prelim_test_data.p
'Submitted batch job 11703397'

#running
name=deepcoder_rl_exp86_0.5_start_holes_0.0001_var_red g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs 10 --use_rl --rl_no_syntax  --load_pretrained_model_path '../deepcoder_temp_0.5_1536186291153/deepcoder_holes.p' --inv_temp 0.5 --train_data 'data/DeepCoder_data/T3_A2_V512_L10_train_perm.txt' --rl_lr 0.0001 --reward_fn exp --sample_fn exp --r_max 8 --num_half_lifes 6 --variance_reduction
'experiments/deepcoder_rl_exp86_0.5_start_holes_0.0001_var_red_1536820812072'
python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_rl_exp86_0.5_start_holes_0.0001_var_red_1536820812072/deepcoder_holes.p'
'results/prelim_results_wdcModel__test50_1536854557.p' #looks pretty sharp #50%
#running


#TRYING FLAT REWARD:
name=deepcoder_rl_flat8_0.5_start_holes_0.0001_var_red g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs 10 --use_rl --rl_no_syntax  --load_pretrained_model_path '../deepcoder_temp_0.5_1536186291153/deepcoder_holes.p' --inv_temp 0.5 --train_data 'data/DeepCoder_data/T3_A2_V512_L10_train_perm.txt' --rl_lr 0.0001 --reward_fn flat --sample_fn flat --r_max 8 --variance_reduction
'experiments/deepcoder_rl_flat8_0.5_start_holes_0.0001_var_red_1536819773837'
python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_rl_flat8_0.5_start_holes_0.0001_var_red_1536819773837/deepcoder_holes.p'
'results/prelim_results_wdcModel__test50_1536854267.p' #46%

name=deepcoder_rl_flat8_0.5_start_holes_0.0001 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs 10 --use_rl --rl_no_syntax  --load_pretrained_model_path '../deepcoder_temp_0.5_1536186291153/deepcoder_holes.p' --inv_temp 0.5 --train_data 'data/DeepCoder_data/T3_A2_V512_L10_train_perm.txt' --rl_lr 0.0001 --reward_fn flat --sample_fn flat --r_max 8 
'experiments/deepcoder_rl_flat8_0.5_start_holes_0.0001_1536820309689'
python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_rl_flat8_0.5_start_holes_0.0001_1536820309689/deepcoder_holes.p'
'results/prelim_results_wdcModel__test50_1536854115.p' #50%



sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_rl_linear8_0.5_start_holes_0.0001_var_red_1536783074744/deepcoder_holes.p'


name=deepcoder_long_temp_0.5 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs 10 --load_pretrained_model_path '../deepcoder_temp_0.5_1536186291153/deepcoder_holes.p' --inv_temp 0.5 --train_data 'data/DeepCoder_data/T3_A2_V512_L10_train_perm.txt'
'experiments/deepcoder_long_temp_0.5_1536954101702'
#WAITING TO RUN:
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_long_temp_0.5_1536954101702/deepcoder_holes.p'
'Submitted batch job 11707360'




sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_temp_0.1_1536186462278/deepcoder_holes.p'

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_temp_0.25_1536186426943/deepcoder_holes.p'
Submitted batch job 11704767 works well

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_temp_0.75_1536186359507/deepcoder_holes.p'
Submitted batch job 11704768 works well 

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_temp_1.0_1536186516665/deepcoder_holes.p'
Submitted batch job 11704769 

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_temp_0.5_1536186291153/deepcoder_holes.p_2.p'
Submitted batch job 11704770 

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_temp_0.5_1536186291153/deepcoder_holes.p_3.p' 
Submitted batch job 11704773 

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_temp_0.5_1536186291153/deepcoder_holes.p_3_iter_0.p' 

#checking luke's hypothesis:

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_temp_0.5_1536186291153/deepcoder_holes.p' --n_samples 1
Submitted batch job 11705248     
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_rl_exp86_0.5_start_holes_0.0001_var_red_1536820812072/deepcoder_holes.p' --n_samples 1
Submitted batch job 11705249




name=deepcoder_512_train_normal g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs 10 --Vrange 512 --load_pretrained_model_path '../deepcoder_pretrained_512_1536675470181/deepcoder_pretrained_512.p_9.p' --inv_temp 0.5
name=deepcoder_512_train_wdc g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs 10 --Vrange 512 --load_pretrained_model_path '../deepcoder_pretrained_512_1536675470181/deepcoder_pretrained_512.p_9.p' --inv_temp 0.5 --use_dc_grammar '../deepcoder_dcModel_512_1536676494070/dc_model_512.p_49.p'




name=deepcoder_128_train_normal_long g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs 10 --load_pretrained_model_path '../deepcoder_pretrained_V128_10_epochs_1536681569098/deepcoder_pretrained.p_9.p' --inv_temp 0.5
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_128_train_normal_long_1537138027905/deepcoder_holes.p' 
Submitted batch job 11721185  
Submitted batch job 11721263    

name=deepcoder_128_train_wdc g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --max_epochs 10 --load_pretrained_model_path '../deepcoder_pretrained_V128_10_epochs_1536681569098/deepcoder_pretrained.p_9.p' --inv_temp 0.5 --use_dc_grammar 'dc_model.p'
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_128_train_wdc_1537138042008/deepcoder_holes.p' 
vim slurm-11721184.out    
Submitted batch job 11721262  



#10 epoch rnn baseline:
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --mdl 14 --model_path 'experiments/deepcoder_pretrained_V128_10_epochs_1536681569098/deepcoder_pretrained.p_9.p' 
vim slurm-11721248.out 

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --mdl 14 --model_path 'experiments/deepcoder_pretrained_V128_10_epochs_1536681569098/deepcoder_pretrained.p_9.p' --n_samples 100
Submitted batch job 11721261 

Super long rnn baseline:
name=deepcoder_long_rnn_base g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --pretrain --load_pretrained_model_path '../deepcoder_pretrained_V128_10_epochs_1536681569098/deepcoder_pretrained.p_9.p' --max_epochs 0 --max_pretrain_epochs 20 --Vrange 128 --save_pretrained_model_path "./deepcoder_rnn_base.p"
vim experiments/deepcoder_long_rnn_base_1537308558965/slurm-11721615.out
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_long_rnn_base_1537308558965/deepcoder_rnn_base.p' 
68% accuracy ... thank the lord

######################DC T4##########

- pretrain on T4:
name=deepcoder_pretrained_T4 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --pretrain --max_epochs 0 --max_pretrain_epochs 1 --max_pretrain_iterations 10000 --Vrange 128 --new --save_pretrained_model_path "./deepcoder_pretrained_T4.p" --train_data 'data/DeepCoder_data/T4_A2_V512_L10_train_perm.txt' --new
vim experiments/deepcoder_pretrained_T4_1537310448749/slurm-11721913.out 

- train dc model on T4:
name=deepcoder_dcModel_T4 g-run sbatch execute_gpu.sh python deepcoder_train_dc_model.py --save_model_path './dc_model_T4.p' --new --train_data 'data/DeepCoder_data/T4_A2_V512_L10_train_perm.txt' --max_epochs 1
vim experiments/deepcoder_dcModel_T4_1537310497320/slurm-11721914.out




######################################################ROBUSTFILL:
name=rb_first_run g-run sbatch execute_gpu.sh python main_supervised_robustfill.py --pretrain
Submitted batch job 11709083

name=rb_first_train_dc_model g-run sbatch execute_gpu.sh python robustfill_train_dc_model.py


name=rb_long_pretrain g-run sbatch execute_gpu.sh python main_supervised_robustfill.py --pretrain --load_pretrained_model_path '../rb_first_run_1537058045390/robustfill_pretrained.p' --max_pretrain_iteration 8000 --max_iteration 8000



sbatch execute_gpu.sh python main_supervised_robustfill.py --pretrain --load_pretrained_model_path 'robustfill_holes.p' --max_iteration 8000


###USING TIMEOUT WITH RB#####
name=rb_timeout_0.5 g-run sbatch execute_gpu.sh python main_supervised_robustfill.py --use_timeout --inv_temp 0.5 --load_pretrained_model_path '../rb_long_pretrain_1537123008935/robustfill_pretrained.p' --max_iteration 5000 --use_dc_grammar '../rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict'
#sbatch execute_gpu.sh python evaluate_robustfill.py --dcModel --model_path [] --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 

name=rb_timeout_1.0 g-run sbatch execute_gpu.sh python main_supervised_robustfill.py --use_timeout --inv_temp 1.0 --load_pretrained_model_path '../rb_long_pretrain_1537123008935/robustfill_pretrained.p' --max_iteration 5000 --use_dc_grammar '../rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict'
#running

name=rb_timeout_0.25 g-run sbatch execute_gpu.sh python main_supervised_robustfill.py --use_timeout --inv_temp 0.25 --load_pretrained_model_path '../rb_long_pretrain_1537123008935/robustfill_pretrained.p' --max_iteration 5000 --use_dc_grammar '../rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict'


#USING TIMEOUT TRAINING DEEPCODER
name=deepcoder_timeout_0.5 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --use_timeout --max_epochs 10 --load_pretrained_model_path '../deepcoder_pretrained_V128_10_epochs_1536681569098/deepcoder_pretrained.p_9.p' --inv_temp 0.5 --use_dc_grammar 'dc_model.p' 
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.5_1537326548865/deepcoder_holes.p' 
Submitted batch job 11728649 - 62% at half trained

name=deepcoder_timeout_1.0 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --use_timeout --max_epochs 10 --load_pretrained_model_path '../deepcoder_pretrained_V128_10_epochs_1536681569098/deepcoder_pretrained.p_9.p' --inv_temp 1.0 --use_dc_grammar 'dc_model.p'
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_1.0_1537326582936/deepcoder_holes.p' 
Submitted batch job 11728650  - 78% at half-trained, wow!! and a nice gradual curve, too!

name=deepcoder_timeout_0.25 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --use_timeout --max_epochs 10 --load_pretrained_model_path '../deepcoder_pretrained_V128_10_epochs_1536681569098/deepcoder_pretrained.p_9.p' --inv_temp 0.25 --use_dc_grammar 'dc_model.p'
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.25_1537327473486//deepcoder_holes.p' 
Submitted batch job 11728651 - 72% at half trained


###ROBUSTFILL preliminary EVALUATION:
sbatch execute_gpu.sh python evaluate_robustfill.py --pretrained --pretrained_model_path './experiments/rb_long_pretrain_1537123008935/robustfill_pretrained.p'
results file saved at rb_results/prelim_results_rnn_baseline__test500_1537201590.p

sbatch execute_gpu.sh python evaluate_robustfill.py --dcModel --dc_baseline --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11717741
results file saved at rb_results/prelim_results_dc_baseline__test500_1537225588.p

sbatch execute_gpu.sh python evaluate_robustfill.py --dcModel --model_path 'experiments/rb_first_run_1537058045390/robustfill_holes.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11717738
results file saved at rb_results/prelim_results_wdcModel__test500_1537207687.p

parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--pretrained_model_path', type=str, default="./robustfill_pretrained.p")
parser.add_argument('--n_test', type=int, default=500)
parser.add_argument('--dcModel', action='store_true')
parser.add_argument('--dc_model_path', type=str, default="./robustfill_dc_model.p")
parser.add_argument('--dc_baseline', action='store_true')
parser.add_argument('--n_samples', type=int, default=30)
parser.add_argument('--mdl', type=int, default=14)  #9
parser.add_argument('--n_examples', type=int, default=4)
parser.add_argument('--Vrange', type=int, default=25)
parser.add_argument('--precomputed_data_file', type=str, default='rb_test_tasks.p')
parser.add_argument('--model_path', type=str, default="./robustfill_holes.p")
parser.add_argument('--max_to_check', type=int, default=5000)


manipulate_results.py --basefile rb_results/prelim_results_rnn_baseline__test500_1537201590.p --resultsfile rb_results/prelim_results_wdcModel__test500_1537207687.p --rb --rnnbase

manipulate_results.py --basefile rb_results/prelim_results_dc_baseline__test500_1537225588.p --resultsfile rb_results/prelim_results_wdcModel__test500_1537207687.p --rb


#FINAL TRAINING:

########
DEEPCODER:

TRAINING:
T3:
-[X] rnn baseline --training --done, and not worth it to run 20 epochs vs 10
-[X] a good RL model - kinda 
-[X] dc baseline
- 3x my model for comparison
	OLD:
	-[ ] 1.0
	-[X] 0.5
	-[ ] 0.25
	NEW:
	-[X] 1.0 --training
	-[X] 0.5 --training
	-[X] 0.25 --training


train 4, test 5  --optional (might be important to show superiority over very well trained rnn here)
currently:
-[X] pretraining RNN --training ep 4000 looks like a good place where its not overfit
-[X] training DCmodel --training iter 5million looks reasonable ... may need a bigger network tho ...
- 3x my model for comparison (decide which based on T3)
	OLD: 
	-[ ] 0.5
	NEW: (use best version)
	-[ ] 1.0
	-[ ] 0.5
	-[ ] 0.25

- bottleneck: training regime ... is it okay to use it?? -- so far it looks very good!!!


TESTING:
-[ ] long tests on actual data
-[ ] varying number of samples
-[ ] the samples vs enum budget frontier graph -- this could be nasty to do



ROBUSTFILL:

TRAINING: 
-[X] rnn baseline
-[ ] a good RL model  maybe not worth --optional
-[X] dc baseline
- 3x my model for comparison (decide which based on T3)
	OLD: 
	-[ ] 1.0
	-[X] 0.5
	-[ ] 0.25
	NEW: (use best)
	-[ ] 1.0 --training
	-[ ] 0.5 --training
	-[ ] 0.25 --training


TESTING:
-[ ] long tests on testing data,
-[ ] use the "real" sygis data to make the correctness graphs Armando wants
-[ ] the samples vs enum budget graph




TODO: 
-[ ] decision on which types of models still to train
-[ ] evaluation: write all of the scripts up for eval and for plotting 
	-[ ] for ease, make eval take a file path as input, so i dont have to look things up
	-[ ] long general runs (DC and RB)
	-[ ] real data graphs (RB)
	-[ ] frontier graphs (DC + RB)?








# - need
# cd $regex_folder
# sbatch gpu_execute.sh python regex_evaluate.py -- 'regex_model' + options 
# regex_results_file = # TODO #put results file here
	
# #baseline evaluation
# cd $regex_folder
# sbatch gpu_execute.sh python regex_evaluate.py -- 'regex_model' + options + --baseline=true or something
# regex_base_results_file = # TODO #put results file here

# #graphing + manupulating results:
# cd $regex_folder

# and the same for DC

# # ran 
# name=deepcoder_rl_temp_0.5 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --inv_temp 0.5 --use_rl --variance_reduction --max_epochs 15
# 'experiments/deepcoder_rl_temp_0.5_1536636279865'
# name=deepcoder_rl_temp_1.0 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --inv_temp 1 --use_rl --variance_reduction --max_epochs 15
# 'experiments/deepcoder_rl_temp_1.0_1536636352841'

# name=deepcoder_rl_temp_0.5_no_syntax g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --inv_temp 0.5 --use_rl --variance_reduction --max_epochs 15 --rl_no_syntax
# 'experiments/deepcoder_rl_temp_0.5_no_syntax_1536636487228'
# name=deepcoder_rl_temp_1.0_no_syntax g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --inv_temp 1 --use_rl --variance_reduction --max_epochs 15 --rl_no_syntax
# 'experiments/deepcoder_rl_temp_1.0_no_syntax_1536636518271'


# name=deepcoder_rl_temp_0.5_boost_obj g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --inv_temp 0.5 --use_rl --variance_reduction --max_epochs 15
# 'experiments/deepcoder_rl_temp_0.5_boost_obj_1536638299528'
# name=deepcoder_rl_temp_1.0_boost_obj  g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --inv_temp 1 --use_rl --variance_reduction --max_epochs 15
# 'experiments/deepcoder_rl_temp_1.0_boost_obj_1536638330476'

# # how was it evaluated? - may need to reeval everything
#python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 10 --model_path 'experiments/deepcoder_temp_0.5_1536186291153/deepcoder_holes.p'                                      

# python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 10 --model_path 'experiments/deepcoder_rl_temp_0.5_1536636279865/deepcoder_holes.p'
# python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 10 --model_path 'experiments/deepcoder_rl_temp_1.0_1536636352841/deepcoder_holes.p'
# python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 10 --model_path 'experiments/deepcoder_rl_temp_0.5_no_syntax_1536636487228/deepcoder_holes.p'
# python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 10 --model_path 'experiments/deepcoder_rl_temp_1.0_no_syntax_1536636518271/deepcoder_holes.p'
# python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 10 --model_path 'experiments/deepcoder_rl_temp_0.5_boost_obj_1536638299528/deepcoder_holes.p'
# python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 10 --model_path 'experiments/deepcoder_rl_temp_1.0_boost_obj_1536638330476/deepcoder_holes.p'

