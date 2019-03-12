#train_and_test.sh
#script to train and test model


if [[ "$@" == "--inner" ]]; then

	which python

	#Only pretrain
	RES_PRE=$(sbatch --parsable -e 'pretrain.out' -o 'pretrain.out' execute_gpu.sh python train/main_supervised_robustfill.py --pretrain --max_pretrain_iteration 4500 --load_pretrained_model_path ../rb_long_pretrain_1537123008935/robustfill_pretrained.p --input_noise)
	#echo "pretraining job: $RES_PRE"

	# train dc_model:
	RES_DC=$(sbatch --parsable -e 'dctrain.out' -o 'dctrain.out' execute_gpu.sh python train/robustfill_train_dc_model.py --max_iteration 1500000 --inv_temp 0.05 --nHoles 3 -k 50 --improved_dc_model --input_noise --load_model_path ../rb_multihole_1experiment_1545242661738/saved_models/algolisp_dc_model.p) #or not.. --use_dc_grammar
 	echo "dc model training job: $RES_DC"

	# train model:
	RES_TRAIN=$(sbatch --parsable --dependency=afterok:$RES_DC -e 'train.out' -o 'train.out' execute_gpu.sh python train/main_supervised_robustfill.py --use_dc_grammar './saved_models/text_dc_model.p' --improved_dc_model --inv_temp 0.25 --nHoles 3 -k 50 --use_timeout --timing --max_iteration 15000 --input_noise --load_trained_model_path ../rb_multihole_1experiment_1545242661738/saved_models/robustfill_pretrained.p)

	# test model
	echo "Eval job:"
	sbatch --dependency=afterok:$RES_TRAIN -e 'eval.out' -o 'eval.out' execute_public_cpu.sh python eval/evaluate_robustfill.py --beam --n_samples 50 --max_to_check 20000 --dcModel --improved_dc_grammar --input_noise

	echo "Eval rnn job:"
	sbatch --dependency=afterok:$RES_PRE -e 'evalrnn.out' -o 'evalrnn.out' execute_public_cpu.sh python eval/evaluate_deepcoder.py  --max_to_check 20000 --dcModel --improved_dc_grammar --model_path "./saved_models/list_pretrained.p" --resultsfile "multihole_list_results_base" --max_to_check 20000 --input_noise
	
	echo "Eval dc baseline job:"
	sbatch --dependency=afterok:$RES_DC -e 'evaldc.out' -o 'evaldc.out' execute_public_cpu.sh python eval/evaluate_deepcoder.py --dc_baseline --resultsfile "multihole_list_results_dc" --max_to_check 20000 --input_noise

else
	#to activate, should properly run:
	echo "running main script at run.txt"
	name=rb_noise g-run bash text_train_and_test.sh --inner > run.txt & #can i do this??
fi


# python train/main_supervised_deepcoder.py --max_epochs 10 --use_dc_grammar './saved_models/list_dc_model.p' --inv_temp 0.25 --nHoles 3 -k 50 --use_timeout --improved_dc_model --load_pretrained_model_path experiments/deepcoder_pretrained_V128_10_epochs_1536681569098/deepcoder_pretrained.p_9.p)




#sbatch -e 'eval.out' -o 'eval.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9807 --only_passable --queue --n_processes 44 --timeout 1000 --max_to_check 20000