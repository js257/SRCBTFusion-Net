# SRCBTFusion-Net
<pre>We will upload all model codes after the paper is accepted.</pre>
# Introduction
Convolutional neural network (CNN) and Transformer-based self-attention models have their advantages in extracting local information and global semantic information, and it is a trend to design a model combining stacked residual convolution blocks (SRCB) and Transformer. How to efficiently integrate the two mechanisms to improve the segmentation effect of remote sensing (RS) images is an urgent problem to be solved. We propose an efficient fusion via SRCB and Transformer (SRCBTFusion-Net) as a new semantic segmentation architecture for RS images. The SRCBTFusion-Net adopts an encoder-decoder structure, and the Transformer is embedded into SRCB to form a double coding structure, then the coding features are up-sampled and fused with multi-scale features of SRCB to form a decoding structure. Firstly, a semantic information enhancement module (SIEM) is proposed to get global clues for enhancing deep semantic information. Then, the relationship guidance module (RGM) is designed and combined to re-encode the up-sampled feature map of the decoder to improve the edge segmentation effect. Secondly, a multi-scale feature aggregation module (MFAM) is developed to enhance the extraction of semantic and contextual information, thus alleviating the loss of image feature information and improving the ability to identify similar categories. The proposed SRCBTFusion-Net 's MIoU results on  the Vaihingen and Potsdam datasets are 1.26% and 3.53% higher than the backbone network, respectively, and are superior to the state-of-the-art methods.
# 1. Data Preparation
## 1.1 Potsdam and Vaihingen Datasets 
<pre>Our divided experimental Vaihingen dataset and Potsdam dataset (https://www.aliyundrive.com/s/VjRwXPLYedt)
Extraction code:l2x4<br>
Then prepare the datasets in the following format for easy use of the code:
├── datasets
    ├── Postdam
    │   ├── origin
    │   ├── train
    │   │   ├── images
    │   │   ├── labels
    │   │   └── train_org.txt
    │   └── val
    │       ├── images
    │       ├── labels
    │       └── val_org.txt
    └── Vaihingen
        ├── origin
        ├── train
        │   ├── images
        │   ├── labels
        │   └── train_org.txt
        └── val
            ├── images
            ├── labels
            └── val_org.txt
</pre>
# 2. Training
## 2.1 Pre-training weight
<pre>If you don't want to train, you can adopt the weights we trained on two datasets (https://pan.baidu.com/s/1VRXZ4uFhGcOZMmexmre4BA)
Extraction code: cfks</pre>
## 2.2 Start training and testing
<pre>python transformerCNN/train.py</pre>
# Requirement
<pre>Python 3.7.0+
Pytorch 1.2.0
CUDA 11.2
tqdm 4.65.0
</pre>
