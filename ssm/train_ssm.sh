## test on small GPU
CUDA_VISIBLE_DEVICES=0 python train_ssm.py --dataset potsdam --ssm-scale 512 --ssm-model hrnet --checkname MAR_16_potsdam_hrnet_512 --batch-size 2 --workers 2 --epochs 200

## potsdam scale1 (size 512)
#CUDA_VISIBLE_DEVICES=0 python train_ssm.py --dataset potsdam --ssm-scale 512 --ssm-model hrnet --checkname MMM_DD_DATASET_MODEL_SCALE --batch-size 14 --workers 2 --epochs 200

## gid scale2 (size 768)
#CUDA_VISIBLE_DEVICES=0 python train_ssm.py --dataset gid --ssm-scale 768 --ssm-model pspnet --checkname MMM_DD_DATASET_MODEL_SCALE --batch-size 10 --workers 2 --epochs 200

## wfv scale3 (size 1024)
#CUDA_VISIBLE_DEVICES=0 python train_ssm.py --dataset wfv --ssm-scale 1024 --ssm-model unet --checkname MMM_DD_DATASET_MODEL_SCALE --batch-size 14 --workers 2 --epochs 500


### for scale4 (size 1280)
#CUDA_VISIBLE_DEVICES=0 python train_ssm.py --dataset wfv --ssm-scale 1280 --ssm-model deeplab --checkname MMM_DD_DATASET_MODEL_SCALE --batch-size 24 --workers 2 --epochs 500