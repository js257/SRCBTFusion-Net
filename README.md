# SRCBTFusion-Net
# Introduction
Convolutional neural network (CNN) and Transformer-based self-attention models have their advantages in extracting local information and global semantic information, and it is a trend to design a model combining stacked residual convolution blocks (SRCB) and Transformer. How to efficiently integrate the two mechanisms to improve the segmentation effect of remote sensing (RS) images is an urgent problem
to be solved. An efficient fusion via SRCB and Transformer (SRCBTFusion-Net) is proposed as a new semantic segmentation architecture for RS images. The SRCBTFusion-Net adopts an encoder-decoder structure, and the Transformer is embedded into SRCB to form a double coding structure, then the coding features are up-sampled and fused with multi-scale features of SRCB to form a decoding structure. Firstly, a semantic
information enhancement module (SIEM) is proposed to get global clues for enhancing deep semantic information. Subsequently, the relationship guidance module (RGM) is incorporated to re-encode the decoder’s upsampled feature maps, enhancing the edge segmentation performance. Secondly, a multipath atrous self-attention module (MASM) is developed to enhance the effective selection and weighting of low-level features, effectively reducing the potential confusion introduced by the skip connections between low-level and high-level features. Finally, a multi-scale feature aggregation module (MFAM) is developed to enhance the extraction of semantic and contextual information, thus alleviating the loss of image feature information and improving the ability to identify similar categories. The proposed SRCBTFusion-Net’s performance on the Vaihingen and Potsdam datasets is superior to the state-of-the-art methods.
# 1. Data Preparation
## 1.1 Potsdam and Vaihingen Datasets 
Our divided experimental Vaihingen dataset and Potsdam dataset (https://www.aliyundrive.com/s/VjRwXPLYedt)<br>
Extraction code:l2x4<br>
Then prepare the datasets in the following format for easy use of the code:
<pre>├── datasets
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
If you don't want to train, you can adopt the weights we trained on two datasets (https://pan.baidu.com/s/1VRXZ4uFhGcOZMmexmre4BA)<br>
Extraction code: cfks
## 2.2 Start training and testing
<pre>python transformerCNN/train.py</pre>
# 3. Result:
Comparison of different methods in performance on Potsdam and Vaihingen Datasets:

  </td>
 </tr>
</tbody></table>

![image](https://github.com/js257/SRCBTFusion-Net/blob/3be7237948769651c2eb4e23246cd6b944ed0fb5/figure/fig13.jpg)<br>
Fig. 1. Examples of semantic segmentation results of different models on Potsdam dataset, the last column shows the predictions of our SRCBTFusion-Net, GT represents real label.<br>

![image](https://github.com/js257/SRCBTFusion-Net/blob/3be7237948769651c2eb4e23246cd6b944ed0fb5/figure/fig14.jpg)<br>
Fig. 2. Examples of semantic segmentation results of different models on Vaihingen dataset, the last column shows the predictions of our SRCBTFusion-Net, GT represents real label.<br>
# If you use our SRCBTFusion-Net, please cite our paper:
<pre>
@ARTICLE{10328787,
  author={Chen, Junsong and Yi, Jizheng and Chen, Aibin and Lin, Hui},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SRCBTFusion-Net: An efficient Fusion Architecture via Stacked Residual Convolution Blocks and Transformer for Remote Sensing Image Semantic Segmentation}, 
  year={2023},
  pages={1-1},
  doi={10.1109/TGRS.2023.3336689}}
</pre>
# Requirement
<pre>Python 3.7.0+
Pytorch 1.8.2
CUDA 12.2
tqdm 4.63.0
numpy 1.21.6
ml-collections
collections
scipy
logging
</pre>
