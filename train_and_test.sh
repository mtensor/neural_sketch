#train_and_test.sh
#script to train and test model


if [[ "$@" == "--inner" ]]; then

	which python

	#Only pretrain
	RES_PRE=$(sbatch --parsable -e 'pretrain.out' -o 'pretrain.out' execute_gpu.sh python train/main_supervised_algolisp.py --pretrain --max_epochs 0 --max_pretrain_epochs 40 --limit_data 0.075 --train_to_convergence --converge_after 7)
	echo "pretraining job: $RES_PRE"


	# train dc_model:
	RES_DC=$(sbatch --parsable -e 'dctrain.out' -o 'dctrain.out' execute_gpu.sh python train/algolisp_train_dc_model.py --limit_data 0.05 --max_epochs 40 --inv_temp 0.075 --nHoles 3 -k 50 --use_dc_grammar)
 	echo "dc model training job: $RES_DC"

 	#SLEEP if not ready
 	#while [ "$(sacct -j $RES_PRE.batch --format State --parsable | tail -n 1)" != "COMPLETED|" ] && [ "$(sacct -j $RES_DC.batch --format State --parsable | tail -n 1)" != "COMPLETED|" ]; do sleep 5; done
	
	# train model:
	RES_TRAIN=$(sbatch --parsable --dependency=afterok:$RES_PRE -e 'train.out' -o 'train.out' execute_gpu.sh python train/main_supervised_algolisp.py --limit_data 0.075 --train_to_convergence --converge_after 7 --max_epochs 40 --use_dc_grammar './saved_models/algolisp_dc_model.p' --inv_temp 0.25 --nHoles 3 -k 50 --use_timeout)


	#while [ "$(sacct -j $RES_TRAIN.batch --format State --parsable | tail -n 1)" != "COMPLETED|" ]; do sleep 5; done

	# test model
	echo "Eval job:"
	sbatch --dependency=afterok:$RES_TRAIN -e 'eval.out' -o 'eval.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --queue --n_processes 44 --timeout 600 --max_to_check 20000

	echo "Eval rnn job:"
	sbatch --dependency=afterok:$RES_PRE -e 'evalrnn.out' -o 'evalrnn.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --model_path "./saved_models/algolisp_pretrained.p" --resultsfile "first_rnn_algolisp_results_base" --queue --n_processes 44 --timeout 600 --max_to_check 20000
	
	echo "Eval dc baseline job:"
	sbatch --dependency=afterok:$RES_DC -e 'evaldc.out' -o 'evaldc.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --dc_baseline --resultsfile "first_rnn_algolisp_results_dc" --queue --n_processes 44 --timeout 600 --max_to_check 20000

else
	#to activate, should properly run:
	echo "running main script at run.txt"
	name=algolisp_limit075_first_run g-run bash train_and_test.sh --inner > run.txt & #can i do this??
fi


#RES_PRE=$(sbatch --parsable -e 'pretrain.out' -o 'pretrain.out' execute_gpu.sh python train/main_supervised_algolisp.py --pretrain --max_epochs 0 --max_pretrain_epochs 15 --limit_data 0.2 --train_to_convergence)

#RES_TRAIN=$(sbatch --parsable --dependency=afterok:$RES_PRE -e 'train.out' -o 'train.out' execute_gpu.sh python train/main_supervised_algolisp.py --limit_data 0.2 --train_to_convergence --max_epochs 15 --use_dc_grammar './saved_models/algolisp_dc_model.p' --inv_temp 0.25 --nHoles 3 -k 50 --use_timeout)

#sbatch --dependency=afterok:$RES_TRAIN -e 'eval.out' -o 'eval.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --queue --n_processes 44 --timeout 600 --max_to_check 20000

#sbatch --dependency=afterok:$RES_PRE -e 'evalrnn.out' -o 'evalrnn.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --model_path "./saved_models/algolisp_pretrained.p" --resultsfile "first_rnn_algolisp_results_base" --queue --n_processes 44 --timeout 600 --max_to_check 20000


#python -i eval/evaluate_algolisp.py --n_test 9807 --only_passable --queue --n_processes 28 --timeout 600 --max_to_check 20000
#
# use --dependency=afterok:jobid[:jobid...] (https://hpc.nih.gov/docs/job_dependencies.html)

#sbatch --parsable -e 'train_early.out' -o 'train_early.out' execute_gpu.sh python train/main_supervised_algolisp.py --filter_depth 1 2 3 4 5 6 7 --max_epochs 7 --use_dc_grammar './saved_models/algolisp_dc_model.p_early_start'

#sbatch  -e 'evalrnn1.out' -o 'evalrnn1.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --model_path "./saved_models/algolisp_pretrained.p" --resultsfile "first_rnn_algolisp_results_base_1"

#sbatch  -e 'evaldc1.out' -o 'evaldc1.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --dc_baseline --resultsfile "first_rnn_algolisp_results_dc_1"

#sbatch -e 'eval1.out' -o 'eval1.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --resultsfile 'first_rnn_algolisp_results_holes1'

#sbatch  -e 'evalrnn2.out' -o 'evalrnn2.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --model_path "./saved_models/algolisp_pretrained.p" --resultsfile "first_rnn_algolisp_results_base_2"


# RES_TRAIN=$(sbatch --parsable -e 'train.out' -o 'train.out' execute_gpu.sh python train/main_supervised_algolisp.py --limit_data 0.10 --train_to_convergence --max_epochs 15 --use_dc_grammar './saved_models/algolisp_dc_model.p' --inv_temp 0.25 --nHoles 3 -k 50 --use_timeout --load_trained_model)


# 	#while [ "$(sacct -j $RES_TRAIN.batch --format State --parsable | tail -n 1)" != "COMPLETED|" ]; do sleep 5; done

# 	# test model
# 	echo "Eval job:"
# 	sbatch --dependency=afterok:$RES_TRAIN -e 'eval.out' -o 'eval.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --queue --n_processes 44 --timeout 600 --max_to_check 20000


# 	sbatch -e 'evaltimeout.out' -o 'evaltimeout.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --queue --n_processes 44 --timeout 600 --max_to_check 20000


	# RES_PRE=$(sbatch --parsable -e 'pretrainlong.out' -o 'pretrainlong.out' execute_gpu.sh python train/main_supervised_algolisp.py --pretrain --max_epochs 0 --max_pretrain_epochs 30 --limit_data 0.10 --train_to_convergence --converge_after 1)
	# echo "pretraining job: $RES_PRE"

	# echo "Eval rnn job:"
	# sbatch --dependency=afterok:$RES_PRE -e 'evalrnnlong.out' -o 'evalrnnlong.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --model_path "./saved_models/algolisp_pretrained.p" --resultsfile "first_rnn_algolisp_results_base_long" --queue --n_processes 44 --timeout 600 --max_to_check 20000
	#Submitted batch job 12505249

#sbatch -e 'evalprelim.out' -o 'evalprelim.out' execute_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --queue --n_processes 44 --timeout 600 --max_to_check 20000 --resultsfile "first_rnn_algolisp_results_prelim"
#12506949

#sbatch --dependency=afterok:$RES_PRE -e 'evalrnnlongqos.out' -o 'evalrnnlongqos.out' execute_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --model_path "./saved_models/algolisp_pretrained.p" --resultsfile "first_rnn_algolisp_results_base_long_qos" --queue --n_processes 44 --timeout 600 --max_to_check 20000


# train model:
# RES_TRAIN_PRELIM=$(sbatch --parsable -e 'trainprelim.out' -o 'trainprelim.out' execute_gpu.sh python train/main_supervised_algolisp.py --limit_data 0.05 --train_to_convergence --converge_after 10 --max_epochs 40 --use_dc_grammar './saved_models/algolisp_dc_model.p' --inv_temp 0.25 --nHoles 3 -k 50 --use_timeout --save_model_path "./saved_models/algolisp_holes_prelim.p")

# # test model
# echo "Eval job:"
# sbatch --dependency=afterok:$RES_TRAIN_PRELIM -e 'evalprelim.out' -o 'evalprelim.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --queue --n_processes 44 --timeout 600 --max_to_check 20000 --model_path "./saved_models/algolisp_holes_prelim.p" --resultsfile "first_rnn_algolisp_results_prelim"

