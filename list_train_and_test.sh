#train_and_test.sh
#script to train and test model


if [[ "$@" == "--inner" ]]; then

	which python

	#Only pretrain
	#RES_PRE=$(sbatch --parsable -e 'pretrain.out' -o 'pretrain.out' execute_gpu.sh python train/main_supervised_deepcoder.py --pretrain --max_epochs 0 --max_pretrain_epochs 10 )
	#echo "pretraining job: $RES_PRE"

	# train dc_model:
	RES_DC=$(sbatch --parsable -e 'dctrain.out' -o 'dctrain.out' execute_gpu.sh python train/deepcoder_train_dc_model.py --max_epochs 1 --max_iterations 1000000 --inv_temp 0.05 --nHoles 3 -k 50 --improved_dc_model --train_data data/DeepCoder_data/T44.txt) #or not.. --use_dc_grammar
 	echo "dc model training job: $RES_DC"

	# train model:
	RES_TRAIN=$(sbatch --parsable --dependency=afterok:$RES_DC -e 'train.out' -o 'train.out' execute_gpu.sh python train/main_supervised_deepcoder.py --max_epochs 1 --max_iterations 16000 --use_dc_model './saved_models/list_dc_model.p' --improved_dc_model --inv_temp 0.25 --nHoles 3 -k 50 --use_timeout --timing  --load_pretrained_model_path ../deepcoder_pretrained_T44_1539221782494/saved_models/deepcoder_pretrained_T44.p --train_data data/DeepCoder_data/T44.txt)

	# test model
	echo "Eval job:"
	sbatch --dependency=afterok:$RES_TRAIN -e 'eval.out' -o 'eval.out' execute_public_cpu.sh python eval/evaluate_deepcoder.py --max_to_check 20000 --dcModel --improved_dc_grammar --precomputed_data_file './data/T44_test_new.p'

	echo "Eval rnn job:"
	#sbatch --dependency=afterok:$RES_PRE -e 'evalrnn.out' -o 'evalrnn.out' execute_public_cpu.sh python eval/evaluate_deepcoder.py  --max_to_check 20000 --dcModel --improved_dc_grammar --model_path "./saved_models/list_pretrained.p" --resultsfile "multihole_list_results_base" --max_to_check 20000
	
	echo "Eval dc baseline job:"
	#sbatch --dependency=afterok:$RES_DC -e 'evaldc.out' -o 'evaldc.out' execute_public_cpu.sh python eval/evaluate_deepcoder.py --dc_baseline --resultsfile "multihole_list_results_dc" --max_to_check 20000

else
	#to activate, should properly run:
	echo "running main script at run.txt"
	name=deepcoder_multihole_T44_1experiment g-run vim & #can i do this??
fi


# python train/main_supervised_deepcoder.py --max_epochs 10 --use_dc_grammar './saved_models/list_dc_model.p' --inv_temp 0.25 --nHoles 3 -k 50 --use_timeout --improved_dc_model --load_pretrained_model_path experiments/deepcoder_pretrained_V128_10_epochs_1536681569098/deepcoder_pretrained.p_9.p)