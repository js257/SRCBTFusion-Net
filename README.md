# SRCBTFusion-Net
We will upload all model codes after the paper is accepted.
# Introduction
Convolutional neural network (CNN) and Transformer-based self-attention models have their advantages in extracting local information and global semantic information, and it is a trend to design a model combining stacked residual convolution blocks (SRCB) and Transformer. How to efficiently integrate the two mechanisms to improve the segmentation effect of remote sensing (RS) images is an urgent problem to be solved. We propose an efficient fusion via SRCB and Transformer (SRCBTFusion-Net) as a new semantic segmentation architecture for RS images. The SRCBTFusion-Net adopts an encoder-decoder structure, and the Transformer is embedded into SRCB to form a double coding structure, then the coding features are up-sampled and fused with multi-scale features of SRCB to form a decoding structure. Firstly, a semantic information enhancement module (SIEM) is proposed to get global clues for enhancing deep semantic information. Then, the relationship guidance module (RGM) is designed and combined to re-encode the up-sampled feature map of the decoder to improve the edge segmentation effect. Secondly, a multi-scale feature aggregation module (MFAM) is developed to enhance the extraction of semantic and contextual information, thus alleviating the loss of image feature information and improving the ability to identify similar categories. The proposed SRCBTFusion-Net 's MIoU results on the Vaihingen and Potsdam datasets are 1.26% and 3.53% higher than the backbone network, respectively, and are superior to the state-of-the-art methods.
# 1. Data Preparation
## 1.1 Potsdam and Vaihingen Datasets 
Our divided experimental Vaihingen dataset and Potsdam dataset (https://www.aliyundrive.com/s/VjRwXPLYedt)<br>
Extraction code:l2x4<br>
├── Postdam<br >
│   ├── origin<br >
│   ├── train<br >
│   └── val<br >
└── val<br >
    ├── origin<br>
    ├── train<br>
    └── val<br>
# 2. Training
## 2.1 Pre-training weight
If you don't want to train, you can adopt the weights we trained on two datasets (https://pan.baidu.com/s/1VRXZ4uFhGcOZMmexmre4BA)<br>
Extraction code: cfks<br>
## 2.2 Start training and testing
python transformerCNN/train.py<br>
# Requirement
Python 3.7.0+<br>
Pytorch 1.2.0<br>
CUDA 11.2<br>
tqdm 4.65.0<br>
