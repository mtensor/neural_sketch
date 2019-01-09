#train_and_test.sh
#script to train and test model


if [[ "$@" == "--inner" ]]; then

	which python

	#Only pretrain
	RES_PRE=$(sbatch --parsable -e 'pretrain.out' -o 'pretrain.out' execute_gpu.sh python train/main_supervised_algolisp.py --seed 43 --pretrain --max_epochs 0 --max_pretrain_epochs 45 --use_dataset_len 16000 --train_to_convergence)
	echo "pretraining job: $RES_PRE"

	# train dc_model:
	RES_DC=$(sbatch --parsable -e 'dctrain.out' -o 'dctrain.out' execute_gpu.sh python train/algolisp_train_dc_model.py --seed 43 --use_dataset_len 16000 --max_epochs 25 --inv_temp 0.05 --nHoles 3 -k 50)
 	echo "dc model training job: $RES_DC"

	# train model:
	RES_TRAIN=$(sbatch --parsable --dependency=afterok:$RES_PRE -e 'train.out' -o 'train.out' execute_gpu.sh python train/main_supervised_algolisp.py --seed 43 --use_dataset_len 16000 --train_to_convergence --max_epochs 45 --use_dc_grammar './saved_models/algolisp_dc_model.p' --inv_temp 0.25 --nHoles 3 -k 50 --use_timeout)


	# # test model
	# echo "Eval job:"
	# sbatch --dependency=afterok:$RES_TRAIN -e 'eval.out' -o 'eval.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9015 --only_passable --queue --n_processes 44 --timeout 600 --max_to_check 20000 --resultsfile "results_model"

	# echo "Eval rnn job:"
	# sbatch --dependency=afterok:$RES_PRE -e 'evalrnn.out' -o 'evalrnn.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9015 --only_passable --model_path "./saved_models/algolisp_pretrained.p" --resultsfile "results_rnn_base" --queue --n_processes 44 --timeout 600 --max_to_check 20000
	
	# echo "Eval dc baseline job:"
	# sbatch --dependency=afterok:$RES_DC -e 'evaldc.out' -o 'evaldc.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9015 --only_passable --dc_baseline --resultsfile "results_dc_base" --queue --n_processes 44 --timeout 600 --max_to_check 20000

	# test model
	echo "Eval job:"
	sbatch --dependency=afterok:$RES_TRAIN -e 'evaldev.out' -o 'evaldev.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --dataset 'dev' --n_test 9807 --only_passable --queue --n_processes 44 --timeout 600 --max_to_check 20000 --resultsfile "results_dev_model"

	echo "Eval rnn job:"
	sbatch --dependency=afterok:$RES_PRE -e 'evaldevrnn.out' -o 'evaldevrnn.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --dataset 'dev' --n_test 9807 --only_passable --model_path "./saved_models/algolisp_pretrained.p" --resultsfile "results_dev_rnn_base" --queue --n_processes 44 --timeout 600 --max_to_check 20000
	
	echo "Eval dc baseline job:"
	sbatch --dependency=afterok:$RES_DC -e 'evaldevdc.out' -o 'evaldevdc.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --dataset 'dev' --n_test 9807 --only_passable --dc_baseline --resultsfile "results_dev_dc_base" --queue --n_processes 44 --timeout 600 --max_to_check 20000


else
	#to activate, should properly run:
	echo "running main script at run.txt"
	name=algolisp_f16000v2 g-run bash train_and_test.sh --inner > run.txt & #can i do this??
fi