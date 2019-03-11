#train_and_test.sh
#script to train and test model


if [[ "$@" == "--inner" ]]; then

	which python

	#Only pretrain
	RES_PRE=$(sbatch --parsable -e 'pretrain.out' -o 'pretrain.out' execute_gpu.sh python train/main_supervised_algolisp.py --pretrain --max_epochs 0 --max_pretrain_epochs 45 --train_to_convergence)
	echo "pretraining job: $RES_PRE"

	# train dc_model:
	RES_DC=$(sbatch --parsable -e 'dctrain.out' -o 'dctrain.out' execute_gpu.sh python train/algolisp_train_dc_model.py --max_epochs 25 --inv_temp 0.05 --nHoles 3 -k 50)
 	echo "dc model training job: $RES_DC"

	# train model:
	RES_TRAIN=$(sbatch --parsable --dependency=afterok:$RES_PRE -e 'train.out' -o 'train.out' execute_gpu.sh python train/main_supervised_algolisp.py --train_to_convergence --max_epochs 45 --use_dc_grammar './saved_models/algolisp_dc_model.p' --inv_temp 0.25 --nHoles 3 -k 50 --use_timeout)

	echo "eval jobs"
	sbatch --dependency=afterok:$RES_TRAIN -e 'finaleval.out' -o 'finaleval.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9967 --mdl 100 --queue --n_processes 44 --timeout 600 --max_to_check 20000 --resultsfile "results_model"

	sbatch --dependency=afterok:$RES_PRE -e 'finalevalrnn.out' -o 'finalevalrnn.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9967 --mdl 100 --model_path "./saved_models/algolisp_pretrained.p" --resultsfile "results_rnn_base" --queue --n_processes 44 --timeout 600 --max_to_check 20000
	
	sbatch --dependency=afterok:$RES_DC -e 'finalevaldc.out' -o 'finalevaldc.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9967 --mdl 100 --dc_baseline --resultsfile "results_dc_base" --queue --n_processes 44 --timeout 600 --max_to_check 20000

	# echo "dev jobs:"
	# # test model
	sbatch --dependency=afterok:$RES_TRAIN -e 'finaldev.out' -o 'finaldev.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --dataset 'dev' --n_test 10819 --mdl 100 --queue --n_processes 44 --timeout 600 --max_to_check 20000 --resultsfile "results_dev_model"

	sbatch --dependency=afterok:$RES_PRE -e 'finaldevrnn.out' -o 'finaldevrnn.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --dataset 'dev' --n_test 10819 --mdl 100 --model_path "./saved_models/algolisp_pretrained.p" --resultsfile "results_dev_rnn_base" --queue --n_processes 44 --timeout 600 --max_to_check 20000
	
	sbatch --dependency=afterok:$RES_DC -e 'finaldevdc.out' -o 'finaldevdc.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --dataset 'dev' --n_test 10819 --mdl 100 --dc_baseline --resultsfile "results_dev_dc_base" --queue --n_processes 44 --timeout 600 --max_to_check 20000


else
	#to activate, should properly run:
	echo "running main script at run.txt"
	name=algolisp_fullnew g-run bash train_and_test_full.sh --inner > run.txt & #can i do this??
fi


# TRAIN_PRE=$(sbatch --parsable -e 'train.out' -o 'train.out' execute_gpu.sh python train/main_supervised_algolisp.py --save_model_path './saved_models/algolisp_holes_prelim.p' --train_to_convergence --max_epochs 45 --use_dc_grammar './saved_models/algolisp_dc_model.p' --inv_temp 0.25 --nHoles 3 -k 50 --use_timeout)
# 12755716
# sbatch --dependency=afterok:$TRAIN_PRE -e 'finalevalprelim.out' -o 'finalevalprelim.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --model_path "./saved_models/algolisp_holes_prelim.p" --n_test 9967 --mdl 100 --queue --n_processes 44 --timeout 600 --max_to_check 20000 --resultsfile "results_modelprelim" 
# Submitted batch job 12755717


#sbatch -e 'evalrnn.out' -o 'evalrnn.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 8995 --only_passable --model_path "./saved_models/algolisp_pretrained.p" --resultsfile "results_rnn_base" --queue --n_processes 44 --timeout 600 --max_to_check 20000

#sbatch -e 'evaldevprelim.out' -o 'evaldevprelim.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --dataset 'dev' --n_test 9807 --only_passable --queue --n_processes 44 --timeout 600 --max_to_check 20000 --resultsfile "results_dev_model_prelim"


 #sbatch -e 'finaleval16kdc.out' -o 'finaleval16kdc.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --dcModel_path ../algolisp_16000v2_1547091411445/saved_models/algolisp_dc_model.p --n_test 9967 --mdl 100 --queue --n_processes 44 --timeout 600 --max_to_check 20000 --resultsfile "results_model16kdc"


#RES_TRAIN=$(sbatch --parsable -e 'train.out' -o 'train.out' execute_gpu.sh python train/main_supervised_algolisp.py --train_to_convergence --max_epochs 45 --use_dc_grammar './saved_models/algolisp_dc_model.p' --inv_temp 0.25 --nHoles 3 -k 50 --use_timeout --load_trained_model --load_trained_model_path './saved_models/algolisp_holes_prelim.p_1.p')
#sbatch --dependency=afterok:$RES_TRAIN -e 'finalevalprelim.out' -o 'finalevalprelim.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9967 --mdl 100 --queue --n_processes 44 --timeout 600 --max_to_check 20000 --resultsfile "results_model" 
#Submitted batch job 12770078

#RES_TRAIN_F=$(sbatch --parsable -e 'trainfast.out' -o 'trainfast.out' execute_gpu.sh python train/main_supervised_algolisp.py --train_to_convergence --max_epochs 45 --use_dc_grammar './saved_models/algolisp_dc_model.p' --inv_temp 0.25 --nHoles 3 -k 50 --use_timeout --load_trained_model --load_trained_model_path './saved_models/algolisp_holes_prelim.p' --save_model_path './saved_models/algolisp_holes_fast.p')
#sbatch --dependency=afterok:$RES_TRAIN_F -e 'finalevalprelimfast.out' -o 'finalevalprelimfast.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9967 --mdl 100 --queue --n_processes 44 --timeout 600 --max_to_check 20000 --resultsfile "results_model_fast" --model_path './saved_models/algolisp_holes_fast.p'
#Submitted batch job 12770080

#sbatch -e 'finalevalprelimr.out' -o 'finalevalprelimr.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --n_test 9967 --mdl 100 --queue --n_processes 44 --timeout 600 --max_to_check 20000 --resultsfile "results_model_prelim_r"
#Submitted batch job 12781202
#sbatch -e 'finalevalprelimf.out' -o 'finalevalprelimf.out' execute_public_cpu.sh python eval/evaluate_algolisp.py --model_path "./saved_models/algolisp_holes_fast.p" --n_test 9967 --mdl 100 --queue --n_processes 44 --timeout 600 --max_to_check 20000 --resultsfile "results_model_prelim_f"
#Submitted batch job 12781203