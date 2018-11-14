#train_and_test.sh
#script to train and test model


if [[ "$@" == "--inner" ]]; then

	which python

	#Only pretrain
	RES_PRE=$(sbatch --parsable -e 'pretrain.out' -o 'pretrain.out' execute_gpu.sh python train/main_supervised_algolisp.py --pretrain --max_epochs 0 --max_pretrain_epochs 7 --filter_depth 1 2 3 4 5 6 7)
	echo "pretraining job: $RES_PRE"

	# train dc_model:
	RES_DC=$(sbatch --parsable -e 'dctrain.out' -o 'dctrain.out' execute_gpu.sh python train/algolisp_train_dc_model.py --filter_depth --filter_depth 1 2 3 4 5 6 7)
 	echo "dc model training job: $RES_DC"

 	#SLEEP if not ready
 	while [ "$(sacct -j $RES_PRE.batch --format State --parsable | tail -n 1)" != "COMPLETED|" ] && [ "$(sacct -j $RES_DC.batch --format State --parsable | tail -n 1)" != "COMPLETED|" ]; do sleep 5; done
	
	# train model:
	RES_TRAIN=$(sbatch --parsable -e 'train.out' -o 'train.out' execute_gpu.sh python train/main_supervised_algolisp.py --filter_depth --filter_depth 1 2 3 4 5 6 7 --max_epochs 7)


	while [ "$(sacct -j $RES_TRAIN.batch --format State --parsable | tail -n 1)" != "COMPLETED|" ]; do sleep 5; done

	# test model
	echo "Eval job:"
	sbatch  -e 'eval.out' -o 'eval.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable

	echo "Eval rnn job:"
	sbatch  -e 'evalrnn.out' -o 'evalrnn.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --model_path "./saved_models/algolisp_pretrained.p" --resultsfile "first_rnn_algolisp_results_base"
	
	echo "Eval dc baseline job:"
	sbatch  -e 'evaldc.out' -o 'evaldc.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --dc_baseline --resultsfile "first_rnn_algolisp_results_dc"

else
	#to activate, should properly run:
	echo "running main script at run.txt"
	name=algolisp_first_all g-run bash train_and_test.sh --inner > run.txt ; #can i do this??
fi
