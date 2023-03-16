# MFVNet
Codes will be made available ASAP.

This repo is the official implementation of "MFVNet: Deep Adaptive Fusion Network with Multiple Field-of-Views for Remote Sensing Image Semantic Segmentation".

################################################ install ####################################################

conda create -n mfvnet python=3.7

conda activate mfvnet

conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch

(or) (conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch)

conda install rasterio tqdm tensorboardX yacs matplotlib

################################################ download data #############################################

(If you wanna use your own dataset, modify the files in the dataloader folder.)


################################################ train #####################################################

cd PATH_TO_YOUR_WORKING_DIRECTORY

git clone https://github.com/weichenrs/MFVNet

cd MFVNet

cd ssm

sh train_ssm.sh

cd ../mfv

sh retrain_mfv.sh
