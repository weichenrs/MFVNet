# MFVNet

This repo contains the supported code and models to reproduce the results of [MFVNet: a deep adaptive fusion network with multiple field-of-views for remote sensing image semantic segmentation](https://link.springer.com/article/10.1007/s11432-022-3599-y).

<!-- ![](figures/mfvnet.jpg) -->
<figure>
  <text-align: center;>
  <img src="./mfvnet.jpg" alt="mfvnet" title="" width=800" height="320" />
</figcaption>
</figure>

## Updates

***03/17/2023*** Models on the [Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx) dataset are released.

***03/16/2023*** Initial commits.

## Results and Models for MFV

### [Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)

| Method | Imp. sur. | Car | Tree | Low veg. | Building | Clutter | mIoU | FWIoU | mF1 | model |  
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MFVNet | 85.2 | 82.2 | 76.0 | 74.9 | 91.4 | 39.2 | 74.8 | 81.5 | 84.3 | [github](https://github.com/weichenrs/MFVNet/releases/download/models/potsdam_mfv.pth.tar)/[google](https://drive.google.com/file/d/12rfEp1bNDdkbrLP-JPnQq7eiFT7EUxFA/view?usp=share_link)/[baidu](https://pan.baidu.com/s/1SMEj9O0uIPiKc-uR0gcgBw?pwd=3y9z) |

## Results and Models for SSM

### [Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)

| Scale | Method | Imp. sur. | Car | Tree | Low veg. | Building | Clutter | mIoU | FWIoU | mF1 | model |  
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| low (512) | UNet | 82.2 | 82.9 | 73.9 | 72.1 | 88.6 | 31.7 | 71.9 | 78.6 | 81.9 | - |
| low (512) | HRNet | 83.0 | 81.3 | 72.7 | 72.5 | 90.0 | 36.2 | 72.6 | 79.2 | 82.7 | - |
| low (512) | PSPNet | 84.0 | 80.5 | 74.7 | 73.4 | 90.5 | 36.9 | 73.3 | 80.2 | 83.2 | [github](https://github.com/weichenrs/MFVNet/releases/download/models/potsdam_s1_psp.pth.tar)/[baidu](https://pan.baidu.com/s/1vIUHiVk17u9EO-rklGTjyw?pwd=e3ie) |
| middle (768) | UNet | 82.3 | 81.5 | 72.6 | 71.2 | 88.6 | 33.1 | 71.6 | 78.3 | 81.8 | - |
| middle (768) | HRNet | 81.4 | 81.0 | 68.6 | 69.6 | 88.6 | 35.1 | 70.7 | 77.5 | 81.0 | - |
| middle (768) | PSPNet | 83.6 | 79.4 | 73.6 | 73.0 | 90.1 | 37.1 | 72.8 | 79.7 | 82.9 | [github](https://github.com/weichenrs/MFVNet/releases/download/models/potsdam_s2_psp.pth.tar)/[baidu](https://pan.baidu.com/s/1Q_bP8pDpPJAzB0CaWj4bpw?pwd=ch7a) |
| high (1024) | UNet | 80.9 | 80.5 | 71.5 | 69.5 | 88.3 | 31.4 | 70.4 | 77.2 | 80.9 | [github](https://github.com/weichenrs/MFVNet/releases/download/models/potsdam_s3_u.pth.tar)/[baidu](https://pan.baidu.com/s/1AVfZq4_hB03XUlGw4dGHGw?pwd=dury) |
| high (1024) | HRNet | 80.4 | 79.7 | 67.6 | 67.8 | 88.5 | 28.3 | 68.7 | 75.9 | 79.5 | - |
| high (1024) | PSPNet | 79.6 | 72.4 | 68.1 | 68.1 | 88.2 | 30.1 | 67.7 | 75.6 | 79.1 | - |

## Usage

### Installation (for cuda10)
```
conda create -n mfvnet python=3.7
conda activate mfvnet
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
conda install rasterio tqdm tensorboardX yacs matplotlib
cd PATH_TO_YOUR_WORKING_DIRECTORY
git clone https://github.com/weichenrs/MFVNet
```

### Installation (for cuda11)
```
conda create -n mfvnet python=3.7
conda activate mfvnet
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
conda install rasterio tqdm tensorboardX yacs matplotlib
cd PATH_TO_YOUR_WORKING_DIRECTORY
git clone https://github.com/weichenrs/MFVNet
```

### Downloading data

We upload the processed data of Potsdam dataset, which can be downloaded via [google](https://drive.google.com/drive/folders/1vRtZgEWY-3Uv1_iPDHoOurlKxtXs1EjP?usp=sharing) or [baidu](https://pan.baidu.com/s/17kmd06zmn-Zvx5MOYLIdUA?pwd=cgc2).
```
cd PATH_TO_YOUR_WORKING_DIRECTORY
cd MFVNet
mkdir data
unzip potsdam.zip
```
You can also download the source data from the offical website of [Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx), [GID](https://x-ytong.github.io/project/GID.html), and [WFV](http://sendimage.whu.edu.cn/en/mfc-validation-data).

**Notes:**

- The data of [GID](https://x-ytong.github.io/project/GID.html) dataset and [WFV](http://sendimage.whu.edu.cn/en/mfc-validation-data) dataset are too large to upload, you need to download and process the source data yourself if you wanna use them for experiments.
- If you wanna use your own dataset, you have to modify the files in the dataloader folder according to your needs.

### MFV Training and testing

```
cd PATH_TO_YOUR_WORKING_DIRECTORY
cd MFVNet
mkdir ssm_models
# you need move the SSM models (your trained models or our pre-trained models) to the ssm_models folder. 
cd ../mfv
sh retrain_mfv.sh
```

### SSM Training and testing (search your own best models on each scale)
```
cd PATH_TO_YOUR_WORKING_DIRECTORY
cd MFVNet
cd ssm
sh train_ssm.sh
```

## Citing MFVNet
```
@article{mfvnet,
  author = {Li Yansheng,Chen Wei,Huang Xin,Gao Zhi,Li Siwei,He Tao,Yongjun Zhang},
  title = {MFVNet: a deep adaptive fusion network with multiple field-of-views for remote sensing image semantic segmentation},
  journal = {SCIENCE CHINA Information Sciences},
  year = {2023},
  url = {https://doi.org/10.1007/s11432-022-3599-y}
}
```

### Acknowledgement
[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)
