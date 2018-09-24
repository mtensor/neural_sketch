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

name=rb_timeout_0.10 g-run sbatch execute_gpu.sh python main_supervised_robustfill.py --use_timeout --inv_temp 0.1 --load_pretrained_model_path '../rb_long_pretrain_1537123008935/robustfill_pretrained.p' --max_iteration 5000 --use_dc_grammar '../rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict'




#USING TIMEOUT TRAINING DEEPCODER
name=deepcoder_timeout_0.5 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --use_timeout --max_epochs 10 --load_pretrained_model_path '../deepcoder_pretrained_V128_10_epochs_1536681569098/deepcoder_pretrained.p_9.p' --inv_temp 0.5 --use_dc_grammar 'dc_model.p' 
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.5_1537326548865/deepcoder_holes.p' 
Submitted batch job 11728649 - 62% at half trained
Submitted batch job 11733931 at 9 epochs - 78% (i must have flipped things)

name=deepcoder_timeout_1.0 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --use_timeout --max_epochs 10 --load_pretrained_model_path '../deepcoder_pretrained_V128_10_epochs_1536681569098/deepcoder_pretrained.p_9.p' --inv_temp 1.0 --use_dc_grammar 'dc_model.p'
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_1.0_1537326582936/deepcoder_holes.p' 
Submitted batch job 11728650  - 78% at half-trained, wow!! and a nice gradual curve, too!
Submitted batch job 11733932 at 9 epochs - 66%

name=deepcoder_timeout_0.25 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --use_timeout --max_epochs 10 --load_pretrained_model_path '../deepcoder_pretrained_V128_10_epochs_1536681569098/deepcoder_pretrained.p_9.p' --inv_temp 0.25 --use_dc_grammar 'dc_model.p'
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 14 --model_path 'experiments/deepcoder_timeout_0.25_1537327473486/deepcoder_holes.p' 
Submitted batch job 11728651 - 72% at half trained
Submitted batch job 11733933 - at 9 epochs 80%

name=deepcoder_timeout_0.10 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --use_timeout --max_epochs 10 --load_pretrained_model_path '../deepcoder_pretrained_V128_10_epochs_1536681569098/deepcoder_pretrained.p_9.p' --inv_temp 0.1 --use_dc_grammar 'dc_model.p'



#####TIMEOUT ON T4:
name=deepcoder_T4_timeout_0.5 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --use_timeout --inv_temp 0.5 --max_epochs 1 --max_iterations 4000 --load_pretrained_model_path '../deepcoder_pretrained_T4_1537310448749/deepcoder_pretrained_T4.p_0_iter_4000.p' --use_dc_grammar '../deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p' --train_data 'data/DeepCoder_data/T4_A2_V512_L10_train_perm.txt'
prelim eval:
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 15 --max_to_check 10000 --model_path 'experiments/deepcoder_T4_timeout_0.5_1537402236402/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11742352 

name=deepcoder_T4_timeout_0.25 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --use_timeout --inv_temp 0.25 --max_epochs 1 --max_iterations 4000 --load_pretrained_model_path '../deepcoder_pretrained_T4_1537310448749/deepcoder_pretrained_T4.p_0_iter_4000.p' --use_dc_grammar '../deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p' --train_data 'data/DeepCoder_data/T4_A2_V512_L10_train_perm.txt'
prelim eval:
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 15 --max_to_check 10000 --model_path 'experiments/deepcoder_T4_timeout_0.25_1537402389185/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11739911 - 14%

increased n_samples:
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 60 --n_test 50 --dcModel --mdl 15 --max_to_check 10000 --model_path 'experiments/deepcoder_T4_timeout_0.25_1537402389185/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11742414 

name=deepcoder_T4_timeout_0.1 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --use_timeout --inv_temp 0.1 --max_epochs 1 --max_iterations 4000 --load_pretrained_model_path '../deepcoder_pretrained_T4_1537310448749/deepcoder_pretrained_T4.p_0_iter_4000.p' --use_dc_grammar '../deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p' --train_data 'data/DeepCoder_data/T4_A2_V512_L10_train_perm.txt'
prelim eval:
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 15 --max_to_check 10000 --model_path 'experiments/deepcoder_T4_timeout_0.1_1537402614870/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11742351 - 16%

increased n_samples:
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 60 --n_test 50 --dcModel --mdl 15 --max_to_check 10000 --model_path 'experiments/deepcoder_T4_timeout_0.1_1537402614870/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11742413

increased max and mdl:
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 60 --n_test 50 --dcModel --mdl 16 --max_to_check 30000 --model_path 'experiments/deepcoder_T4_timeout_0.1_1537402614870/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11744672 

increased max:
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_samples 60 --n_test 50 --dcModel --mdl 15 --max_to_check 30000 --model_path 'experiments/deepcoder_T4_timeout_0.1_1537402614870/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11746879

name=deepcoder_T4_timeout_0.05 g-run sbatch execute_gpu.sh python main_supervised_deepcoder.py --use_timeout --inv_temp 0.05 --max_epochs 1 --max_iterations 4000 --load_pretrained_model_path '../deepcoder_pretrained_T4_1537310448749/deepcoder_pretrained_T4.p_0_iter_4000.p' --use_dc_grammar '../deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p' --train_data 'data/DeepCoder_data/T4_A2_V512_L10_train_perm.txt'
prelim eval:
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 15 --max_to_check 10000 --model_path 'experiments/deepcoder_T4_timeout_0.05_1537402842504/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11742402

#T4 deepcoder baseline
sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 15 --max_to_check 10000 --dc_baseline --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11739877  sketches are often found after like 2k+ search .... 

sbatch execute_gpu.sh python evaluate_deepcoder.py --n_test 50 --dcModel --mdl 16 --max_to_check 30000 --dc_baseline --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11744673



###ROBUSTFILL preliminary EVALUATION:
sbatch execute_gpu.sh python evaluate_robustfill.py --pretrained --pretrained_model_path './experiments/rb_long_pretrain_1537123008935/robustfill_pretrained.p'
results file saved at rb_results/prelim_results_rnn_baseline__test500_1537201590.p

sbatch execute_gpu.sh python evaluate_robustfill.py --dcModel --dc_baseline --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11717741
results file saved at rb_results/prelim_results_dc_baseline__test500_1537225588.p

sbatch execute_gpu.sh python evaluate_robustfill.py --dcModel --model_path 'experiments/rb_first_run_1537058045390/robustfill_holes.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11717738
results file saved at rb_results/prelim_results_wdcModel__test500_1537207687.p



#######robustfill final evaluation:#########
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





sbatch execute_gpu.sh python evaluate_robustfill.py --test_generalization --resultsfile 'rb_timeout1test' --precomputed_data_file 'rb_test_tasks.p' --dcModel --model_path 'experiments/rb_timeout_1.0_1537385508985/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11750632
sbatch execute_gpu.sh python evaluate_robustfill.py --test_generalization --resultsfile 'rb_timeout.5test' --precomputed_data_file 'rb_test_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.5_1537385713113/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11750633 
sbatch execute_gpu.sh python evaluate_robustfill.py --test_generalization --resultsfile 'rb_timeout.25test' --precomputed_data_file 'rb_test_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.25_1537385744004/robustfill_holes.p_iter_4000.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11750634    
sbatch execute_gpu.sh python evaluate_robustfill.py --test_generalization --resultsfile 'rb_timeout.1test' --precomputed_data_file 'rb_test_tasks.p' --dcModel --model_path 'experiments/rb_timeout_0.10_1537401029301/robustfill_holes.p_iter_3600.p' --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11750635 
sbatch execute_gpu.sh python evaluate_robustfill.py --test_generalization --resultsfile 'rb_rnnbase_test' --precomputed_data_file 'rb_test_tasks.p' --pretrained --pretrained_model_path './experiments/rb_long_pretrain_1537123008935/robustfill_pretrained.p'
Submitted batch job 11750636
sbatch execute_gpu.sh python evaluate_robustfill.py --test_generalization --resultsfile 'rb_dc_base_test' --precomputed_data_file 'rb_test_tasks.p' --dcModel --dc_baseline --dc_model_path 'experiments/rb_first_train_dc_model_1537064318549/rb_dc_model.pstate_dict' 
Submitted batch job 11750637



'experiments/deepcoder_timeout_0.5_1537326548865/deepcoder_holes.p' 

'experiments/deepcoder_timeout_1.0_1537326582936/deepcoder_holes.p' 

'experiments/deepcoder_timeout_0.25_1537327473486/deepcoder_holes.p' 

'experiments/deepcoder_timeout_0.10_1537400900116/deepcoder_holes.p'


######DC T3 final evaluation#####
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



###t3 very long eval###
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_0.1_long' --max_to_check 20000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_0.10_1537400900116/deepcoder_holes.p'
Submitted batch job 11774744
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_0.25_long' --max_to_check 20000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_0.25_1537327473486/deepcoder_holes.p' 
Submitted batch job 11774745
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_0.5_long' --max_to_check 20000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_0.5_1537326548865/deepcoder_holes.p' 
 Submitted batch job 11774746
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_1_long' --max_to_check 20000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_1.0_1537326582936/deepcoder_holes.p'
Submitted batch job 11774747
#sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_rl_long' --max_to_check 20000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_rl_exp86_0.5_start_holes_0.0001_var_red_1536820812072/deepcoder_holes.p'

sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_rnn_long' --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 14 --model_path 'experiments/deepcoder_long_rnn_base_1537308558965/deepcoder_rnn_base.p' 
Submitted batch job 11774750

#dc baseline 
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_dcbase_long' --max_to_check 20000 --dc_baseline --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16
submitted 11774748


##########t3 100k run###########
NEED TO RUN!!!

sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_0.25_100k' --max_to_check 100000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_0.25_1537327473486/deepcoder_holes.p' 

sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_0.5_100k' --max_to_check 100000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_0.5_1537326548865/deepcoder_holes.p' 

sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_timeout_1_100k' --max_to_check 100000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_timeout_1.0_1537326582936/deepcoder_holes.p'

#sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_rl_long' --max_to_check 20000 --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 16 --model_path 'experiments/deepcoder_rl_exp86_0.5_start_holes_0.0001_var_red_1536820812072/deepcoder_holes.p'

#dc baseline 
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_dcbase_100k' --max_to_check 100000 --dc_baseline --precomputed_data_file 'data/prelim_test_data.p' --n_test 100 --dcModel --mdl 18




######DC T4 final eval########
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T4_0.5' --n_samples 100 --n_test 100 --dcModel --mdl 15 --max_to_check 40000 --model_path 'experiments/deepcoder_T4_timeout_0.5_1537402236402/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11770251 
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T4_0.25' --n_samples 100 --n_test 100 --dcModel --mdl 15 --max_to_check 40000 --model_path 'experiments/deepcoder_T4_timeout_0.25_1537402389185/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11770252 
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T4_0.1' --n_samples 100 --n_test 100 --dcModel --mdl 15 --max_to_check 40000 --model_path 'experiments/deepcoder_T4_timeout_0.1_1537402614870/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11770253 
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T4_0.05' --n_samples 100 --n_test 100 --dcModel --mdl 15 --max_to_check 40000 --model_path 'experiments/deepcoder_T4_timeout_0.05_1537402842504/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11770254 
#deepcoder baseline
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T4_dc_base' --n_test 100 --dcModel --mdl 15 --max_to_check 40000 --dc_baseline --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11770255 
#rnn baseline
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T4_rnn_base' --n_samples 100 --n_test 100 --dcModel --mdl 15 --max_to_check 40000 --dc_baseline --precomputed_data_file 'data/prelim_test_data_T5.p' 
Submitted batch job 11770256
NEED TO REDO!!!!!!


#longer run, take 2:
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T4_0.5_100k' --n_samples 100 --n_test 96 --dcModel --mdl 16 --max_to_check 100000 --model_path 'experiments/deepcoder_T4_timeout_0.5_1537402236402/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11774737
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T4_0.25_100k' --n_samples 100 --n_test 96 --dcModel --mdl 16 --max_to_check 100000 --model_path 'experiments/deepcoder_T4_timeout_0.25_1537402389185/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11774738
#deepcoder baseline
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T4_dc_base_100k' --n_test 100 --dcModel --mdl 16 --max_to_check 100000 --dc_baseline --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11774739
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T4_0.1_100k' --n_samples 100 --n_test 96 --dcModel --mdl 16 --max_to_check 100000 --model_path 'experiments/deepcoder_T4_timeout_0.1_1537402614870/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
 Submitted batch job 11774752
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T4_0.05_100k' --n_samples 100 --n_test 96 --dcModel --mdl 16 --max_to_check 100000 --model_path 'experiments/deepcoder_T4_timeout_0.05_1537402842504/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11774753


sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T4_rnn_base_beam_100k' --beam --n_samples 100 --n_test 96 --dcModel --mdl 15 --max_to_check 200 --precomputed_data_file 'data/prelim_test_data_T5.p' --model_path 'experiments/deepcoder_pretrained_T4_1537310448749/deepcoder_pretrained_T4.p_0_iter_4000.p'
REDOING!!!!!! remember its called 100k
Submitted batch job 11788562

sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T4_0.5_200k' --n_samples 100 --n_test 96 --dcModel --mdl 16 --max_to_check 200000 --model_path 'experiments/deepcoder_T4_timeout_0.5_1537402236402/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11774740
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T4_0.25_200k' --n_samples 100 --n_test 96 --dcModel --mdl 16 --max_to_check 200000 --model_path 'experiments/deepcoder_T4_timeout_0.25_1537402389185/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11774741

sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T4_0.1_200k' --n_samples 100 --n_test 96 --dcModel --mdl 16 --max_to_check 200000 --model_path 'experiments/deepcoder_T4_timeout_0.1_1537402614870/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11774754
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T4_0.05_200k' --n_samples 100 --n_test 96 --dcModel --mdl 16 --max_to_check 200000 --model_path 'experiments/deepcoder_T4_timeout_0.05_1537402842504/deepcoder_holes.p' --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11774755

#deepcoder baseline
sbatch execute_gpu.sh python evaluate_deepcoder.py --resultsfile 'dc_T4_dc_base_200k' --n_test 100 --dcModel --mdl 16 --max_to_check 200000 --dc_baseline --precomputed_data_file 'data/prelim_test_data_T5.p' --dcModel_path 'experiments/deepcoder_dcModel_T4_1537310497320/dc_model_T4.p_0_iter_5400000.p'
Submitted batch job 11774742



#finish this
sbatch execute_gpu.sh python make_final_plots.py --filename 'dc_T3' --file_list  'results/dc_timeout_0.1.p' 'results/dc_timeout_0.25.p' 'results/dc_timeout_0.5.p' 'results/dc_timeout_1.p' 'results/dc_rl.p' 'results/dc_rnn.p' 'results/dc_dcbase.p'  --legend_list "Flexible_neural_sketch,_decay=0.1_(ours)" "Flexible_neural_sketch,_decay=0.25_(ours)" "Flexible_neural_sketch,_decay=0.5_(ours)" "Flexible_neural_sketch,_decay=1_(ours)" "Flexible_neural_sketch,_RL_training_(ours)" "RNN_baseline" "Deepcoder_baseline"

#final good stuff 
sbatch execute_gpu.sh python make_final_plots.py --filename 'dc_T3' --file_list 'results/dc_timeout_0.25.p' 'results/dc_timeout_0.5.p' 'results/dc_timeout_1.p' 'results/dc_rl.p' 'results/dc_rnn.p' 'results/dc_dcbase.p'  --legend_list "Flexible_neural_sketch,_decay=0.25_(ours)" "Flexible_neural_sketch,_decay=0.5_(ours)" "Flexible_neural_sketch,_decay=1_(ours)" "Flexible_neural_sketch,_RL_training_(ours)" "RNN_baseline" "Deepcoder_baseline"



sbatch execute_gpu.sh python make_final_plots.py --robustfill --filename 'rb_challenge' --file_list 'rb_results/rb_timeout.1challenge.p' 'rb_results/rb_timeout.25challenge.p' 'rb_results/rb_timeout.5challenge.p' 'rb_results/rb_timeout1challenge.p' 'rb_results/rb_rnnbase_challenge.p' 'rb_results/rb_dc_base_challenge.p' --legend_list "Flexible_neural_sketch,_decay=0.1_(ours)" "Flexible_neural_sketch,_decay=0.25_(ours)" "Flexible_neural_sketch,_decay=0.5_(ours)" "Flexible_neural_sketch,_decay=1_(ours)" "RNN_baseline" "Deepcoder_baseline"

sbatch execute_gpu.sh python make_final_plots.py --robustfill --filename 'rb_challenge' --file_list 'rb_results/rb_timeout.1challenge.p' 'rb_results/rb_timeout.25challenge.p' 'rb_results/rb_timeout1challenge.p' 'rb_results/rb_rnnbase_challenge.p' 'rb_results/rb_dc_base_challenge.p' --legend_list "Flexible_neural_sketch,_decay=0.1_(ours)" "Flexible_neural_sketch,_decay=0.25_(ours)" "Flexible_neural_sketch,_decay=1_(ours)" "RNN_baseline" "Deepcoder_baseline"


sbatch execute_gpu.sh python make_final_plots.py --robustfill --filename 'rb_challenge100' --file_list 'rb_results/rb_timeout.1challenge100.p' 'rb_results/rb_timeout.25challenge100.p' 'rb_results/rb_timeout1challenge100.p' 'rb_results/rb_rnnbase_challenge100.p' 'rb_results/rb_dc_base_challenge.p' --legend_list "Flexible_neural_sketch,_decay=0.1_(ours)" "Flexible_neural_sketch,_decay=0.25_(ours)" "Flexible_neural_sketch,_decay=1_(ours)" "RNN_baseline" "Deepcoder_baseline"


#generalization:

sbatch execute_gpu.sh python make_final_plots.py --generalization --robustfill --filename 'rb_challenge_gen' --file_list 'rb_results/rb_timeout.1challenge.p' 'rb_results/rb_timeout.25challenge.p' 'rb_results/rb_timeout1challenge.p' 'rb_results/rb_rnnbase_challenge.p' 'rb_results/rb_dc_base_challenge.p' --legend_list "Flexible_neural_sketch,_decay=0.1_(ours)" "Flexible_neural_sketch,_decay=0.25_(ours)" "Flexible_neural_sketch,_decay=1_(ours)" "RNN_baseline" "Deepcoder_baseline"


make_final_plots.py --filename 'dc_T4_prelim' --file_list 'results/dc_T4_0.5_100k.p' 'results/dc_T4_dc_base_100k.p'  --legend_list "Flexible_neural_sketch,_decay=0.5_(ours)" "Deepcoder_baseline"


manipulate_results.py --basefile rb_results/prelim_results_rnn_baseline__test500_1537201590.p --resultsfile rb_results/prelim_results_wdcModel__test500_1537207687.p --rb --rnnbase

manipulate_results.py --basefile rb_results/prelim_results_dc_baseline__test500_1537225588.p --resultsfile rb_results/prelim_results_wdcModel__test500_1537207687.p --rb


#FINAL TRAINING:

############USING BEAM SEARCH#########


### beams on robustfill:

sbatch execute_gpu.sh python evaluate_robustfill.py --test_generalization --beam --n_samples 50 --resultsfile 'rb_rnnbase_challenge_beam_50' --precomputed_data_file 'rb_challenge_tasks.p' --pretrained --pretrained_model_path './experiments/rb_long_pretrain_1537123008935/robustfill_pretrained.p'
Submitted batch job 11778585 - cancelled
faster version: 11778619
sbatch execute_gpu.sh python evaluate_robustfill.py --test_generalization --beam --n_samples 100 --resultsfile 'rb_rnnbase_challenge_beam_100' --precomputed_data_file 'rb_challenge_tasks.p' --pretrained --pretrained_model_path './experiments/rb_long_pretrain_1537123008935/robustfill_pretrained.p'
Submitted batch job 11778626
sbatch execute_gpu.sh python evaluate_robustfill.py --test_generalization --beam --n_samples 150 --resultsfile 'rb_rnnbase_challenge_beam_150' --precomputed_data_file 'rb_challenge_tasks.p' --pretrained --pretrained_model_path './experiments/rb_long_pretrain_1537123008935/robustfill_pretrained.p'
Submitted batch job 11778633




sbatch execute_gpu.sh python make_final_plots.py --filename 'dc_T3_long' --file_list 'results/dc_timeout_0.25_long.p' 'results/dc_timeout_0.5_long.p' 'results/dc_timeout_1_long.p' 'results/dc_rnn_long.p' 'results/dc_dcbase_long.p'  --legend_list "Flexible_neural_sketch,_decay=0.25_(ours)" "Flexible_neural_sketch,_decay=0.5_(ours)" "Flexible_neural_sketch,_decay=1_(ours)" "RNN_baseline" "Deepcoder_baseline"




########
DEEPCODER:

TRAINING:
T3:
-[X] rnn baseline --done, and not worth it to run 20 epochs vs 10
-[X] a good RL model - kinda 
-[X] dc baseline
- 3x my model for comparison
	OLD:
	-[ ] 1.0
	-[X] 0.5
	-[ ] 0.25
	NEW:
	-[X] 1.0 --done
	-[X] 0.5 --done
	-[X] 0.25 --done
	-[X] 0.1 --done

train 4, test 5  --optional (might be important to show superiority over very well trained rnn here)
currently:
-[X] pretraining RNN 'deepcoder_pretrained_T4.p_0_iter_4000.p' ep 4000 looks like a good place where its not overfit
-[X] training DCmodel 'dc_model_T4.p_0_iter_5400000.p' iter 5million looks reasonable ... may need a bigger network tho ...
- 4x my model for comparison (decide which based on T3)
	NEW: (use best version)
	-[ ] 0.5 --training
	-[ ] 0.25 --training
	-[ ] 0.1 --training
	-[ ] 0.05 --training

TESTING:
-[ ] long tests on actual data
-[ ] varying number of samples
-[ ] the samples vs enum budget frontier graph -- this could be nasty to do

EVALUTATION

for dc 3: 
[ ] evaluate rnn baseline (vary sample num)
[ ] evaluate rl model
[ ] evalute dc baseline 
[ ] evaluate sketch model 4x (vary sample num)

for dc 5:
[ ] evaluate rnn baseline (vary sample num)
[ ] evalute dc baseline 
[ ] evaluate best sketch model (vary sample num)



ROBUSTFILL:

TRAINING: 
-[X] rnn baseline
-[ ] a good RL model  maybe not worth --optional
-[X] dc baseline
- 4x my model for comparison 
	OLD: 
	-[X] 0.5
	NEW: (use best)
	-[ ] 1.0 --training
	-[ ] 0.5 --training
	-[ ] 0.25 --training
	-[ ] 0.1 --training


TESTING:
-[ ] long tests on testing data,
-[ ] use the "real" sygis data to make the correctness graphs Armando wants
-[ ] the samples vs enum budget graph

[ ] evaluate rnn baseline (vary sample num)
[ ] evalute dc baseline 
[ ] evaluate sketch model 4x (vary sample num)

--test_generalization flag can be used for sygis data testing, still need to write plotter to plot g_hit/hit as function of enum budget


