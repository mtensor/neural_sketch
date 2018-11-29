#train_and_test.sh
#script to train and test model


if [[ "$@" == "--inner" ]]; then

	which python

	#Only pretrain
	#RES_PRE=$(sbatch --parsable -e 'pretrain.out' -o 'pretrain.out' execute_gpu.sh python train/main_supervised_algolisp.py --pretrain --max_epochs 0 --max_pretrain_epochs 7 --filter_depth 1 2 3 4 5 6 7)
	#echo "pretraining job: $RES_PRE"

	# train dc_model:
	RES_DC=$(sbatch --parsable -e 'dctrain.out' -o 'dctrain.out' execute_gpu.sh python train/algolisp_train_dc_model.py --filter_depth 1 2 3 4 5 --max_epochs 14 --inv_temp 0.01 --nHoles 3 -k 100 --use_dc_grammar)
 	echo "dc model training job: $RES_DC"

 	#SLEEP if not ready
 	#while [ "$(sacct -j $RES_PRE.batch --format State --parsable | tail -n 1)" != "COMPLETED|" ] && [ "$(sacct -j $RES_DC.batch --format State --parsable | tail -n 1)" != "COMPLETED|" ]; do sleep 5; done
	
	# train model:
	RES_TRAIN=$(sbatch --parsable --dependency=afterok:$RES_DC -e 'train.out' -o 'train.out' execute_gpu.sh python train/main_supervised_algolisp.py --filter_depth 1 2 3 4 5 --max_epochs 14 --use_dc_grammar './saved_models/algolisp_dc_model.p' --inv_temp 0.25 --nHoles 3 -k 100 --load_pretrained_model_path '../algolisp_first_all_1542668797877/saved_models/algolisp_pretrained.p')


	#while [ "$(sacct -j $RES_TRAIN.batch --format State --parsable | tail -n 1)" != "COMPLETED|" ]; do sleep 5; done

	# test model
	echo "Eval job:"
	sbatch --dependency=afterok:$RES_TRAIN -e 'eval.out' -o 'eval.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --queue --n_processes 44 --timeout 600 --max_to_check 20000

	#echo "Eval rnn job:"
	#sbatch --dependency=afterok:$RES_PRE -e 'evalrnn.out' -o 'evalrnn.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --model_path "./saved_models/algolisp_pretrained.p" --resultsfile "first_rnn_algolisp_results_base" --queue --n_processes 44 --timeout 600 --max_to_check 20000
	
	echo "Eval dc baseline job:"
	sbatch --dependency=afterok:$RES_DC -e 'evaldc.out' -o 'evaldc.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --dc_baseline --resultsfile "first_rnn_algolisp_results_dc" --queue --n_processes 44 --timeout 600 --max_to_check 20000

else
	#to activate, should properly run:
	echo "running main script at run.txt"
	name=algolisp_filter5_k100_t01 g-run bash train_and_test.sh --inner > run.txt & #can i do this??
fi


# use --dependency=afterok:jobid[:jobid...] (https://hpc.nih.gov/docs/job_dependencies.html)

#sbatch --parsable -e 'train_early.out' -o 'train_early.out' execute_gpu.sh python train/main_supervised_algolisp.py --filter_depth 1 2 3 4 5 6 7 --max_epochs 7 --use_dc_grammar './saved_models/algolisp_dc_model.p_early_start'

#sbatch  -e 'evalrnn1.out' -o 'evalrnn1.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --model_path "./saved_models/algolisp_pretrained.p" --resultsfile "first_rnn_algolisp_results_base_1"

#sbatch  -e 'evaldc1.out' -o 'evaldc1.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --dc_baseline --resultsfile "first_rnn_algolisp_results_dc_1"

#sbatch -e 'eval1.out' -o 'eval1.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --resultsfile 'first_rnn_algolisp_results_holes1'

#sbatch  -e 'evalrnn2.out' -o 'evalrnn2.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --model_path "./saved_models/algolisp_pretrained.p" --resultsfile "first_rnn_algolisp_results_base_2"

