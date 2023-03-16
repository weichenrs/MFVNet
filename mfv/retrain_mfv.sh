#gid
#CUDA_VISIBLE_DEVICES=0 python retrain_mfv.py --dataset gid --checkname MMM_DD_DATASET_MODEL_SCALE --batch-size 2 --workers 2 --epochs 20

#wfv
#CUDA_VISIBLE_DEVICES=0 python retrain_mfv.py --dataset wfv --checkname MMM_DD_DATASET_MODEL_SCALE --batch-size 2 --workers 2 --epochs 20

#potsdam
#CUDA_VISIBLE_DEVICES=0 python retrain_mfv.py --dataset potsdam --checkname MMM_DD_DATASET_MODEL_SCALE --batch-size 2 --workers 2 --epochs 50
#CUDA_VISIBLE_DEVICES=0 python retrain_mfv.py --dataset potsdam --checkname MMM_DD_DATASET_MODEL_SCALE --batch-size 2 --workers 2 --epochs 100