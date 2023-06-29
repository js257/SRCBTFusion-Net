# SRCBTFusion-Net
<pre>We will upload all model codes after the paper is accepted.</pre>
# Introduction
Convolutional neural network (CNN) and Transformer-based self-attention models have their advantages in extracting local information and global semantic information, and it is a trend to design a model combining stacked residual convolution blocks (SRCB) and Transformer. How to efficiently integrate the two mechanisms to improve the segmentation effect of remote sensing (RS) images is an urgent problem to be solved. We propose an efficient fusion via SRCB and Transformer (SRCBTFusion-Net) as a new semantic segmentation architecture for RS images. The SRCBTFusion-Net adopts an encoder-decoder structure, and the Transformer is embedded into SRCB to form a double coding structure, then the coding features are up-sampled and fused with multi-scale features of SRCB to form a decoding structure. Firstly, a semantic information enhancement module (SIEM) is proposed to get global clues for enhancing deep semantic information. Then, the relationship guidance module (RGM) is designed and combined to re-encode the up-sampled feature map of the decoder to improve the edge segmentation effect. Secondly, a multi-scale feature aggregation module (MFAM) is developed to enhance the extraction of semantic and contextual information, thus alleviating the loss of image feature information and improving the ability to identify similar categories. The proposed SRCBTFusion-Net 's MIoU results on  the Vaihingen and Potsdam datasets are 1.26% and 3.53% higher than the backbone network, respectively, and are superior to the state-of-the-art methods.
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
# Result:
Comparison of different methods in performance on Potsdam and Vaihingen Datasets:
<table class=MsoTableGrid border=1 cellspacing=0 cellpadding=0 width=482
 style='width:361.5pt;border-collapse:collapse;border:none;mso-border-alt:solid windowtext .5pt;
 mso-yfti-tbllook:1184;mso-padding-alt:0cm .3pt 0cm .3pt'>
 <tr style='mso-yfti-irow:0;mso-yfti-firstrow:yes;height:1.0pt'>
  <td width=104 rowspan=2 style='width:78.0pt;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:double windowtext 1.5pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:double windowtext 1.5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>Method<o:p></o:p></span></b></p>
  </td>
  <td width=76 rowspan=2 style='width:2.0cm;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:double windowtext 1.5pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  mso-border-bottom-alt:double windowtext 1.5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>Parameters
  (M)<o:p></o:p></span></b></p>
  </td>
  <td width=76 rowspan=2 style='width:2.0cm;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:double windowtext 1.5pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  mso-border-bottom-alt:double windowtext 1.5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>Speed
  (FPS)<o:p></o:p></span></b></p>
  </td>
  <td width=66 rowspan=2 style='width:49.6pt;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:double windowtext 1.5pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  mso-border-bottom-alt:double windowtext 1.5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>Flops
  (G)<o:p></o:p></span></b></p>
  </td>
  <td width=85 style='width:63.8pt;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  mso-border-left-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm .3pt 0cm .3pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;color:black;mso-themecolor:text1;mso-font-kerning:
  0pt;mso-fareast-language:EN-US'>Potsdam<span style='mso-spacerun:yes'>&nbsp;
  </span></span></b><b><span lang=EN-US style='font-size:8.0pt;font-family:
  "Times New Roman",serif;mso-fareast-font-family:宋体;mso-font-kerning:0pt;
  mso-fareast-language:EN-US'><o:p></o:p></span></b></p>
  </td>
  <td width=76 style='width:2.0cm;border:solid windowtext 1.0pt;border-right:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>Vaihingen<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:1;height:1.0pt'>
  <td width=85 style='width:63.8pt;border-top:none;border-left:none;border-bottom:
  double windowtext 1.5pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;mso-border-bottom-alt:double windowtext 1.5pt;
  padding:0cm .3pt 0cm .3pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><b><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>MIoU</span></b></span><b><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>
  (%) <o:p></o:p></span></b></p>
  </td>
  <td width=76 style='width:2.0cm;border:none;border-bottom:double windowtext 1.5pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  padding:0cm .3pt 0cm .3pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><b><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>MIoU</span></b></span><b><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>
  (%)<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:2;height:1.0pt'>
  <td width=104 style='width:78.0pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:double windowtext 1.5pt;mso-border-top-alt:double windowtext 1.5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>TransUNet</span></span><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'><o:p></o:p></span></p>
  </td>
  <td width=76 style='width:2.0cm;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:double windowtext 1.5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:double windowtext 1.5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>76.77<o:p></o:p></span></p>
  </td>
  <td width=76 style='width:2.0cm;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:double windowtext 1.5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:double windowtext 1.5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>19<o:p></o:p></span></p>
  </td>
  <td width=66 style='width:49.6pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:double windowtext 1.5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:double windowtext 1.5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>15.51<o:p></o:p></span></p>
  </td>
  <td width=85 style='width:63.8pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:double windowtext 1.5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:double windowtext 1.5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-font-kerning:
  0pt;mso-fareast-language:EN-US'>76.86</span><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'><o:p></o:p></span></p>
  </td>
  <td width=76 style='width:2.0cm;border:none;mso-border-top-alt:double windowtext 1.5pt;
  mso-border-left-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>74.30<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:3;height:1.0pt'>
  <td width=104 style='width:78.0pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>ABCNet</span></span><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'><o:p></o:p></span></p>
  </td>
  <td width=76 style='width:2.0cm;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>28.57<o:p></o:p></span></p>
  </td>
  <td width=76 style='width:2.0cm;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>26<o:p></o:p></span></p>
  </td>
  <td width=66 style='width:49.6pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>7.24<o:p></o:p></span></p>
  </td>
  <td width=85 style='width:63.8pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-font-kerning:
  0pt;mso-fareast-language:EN-US'>74.89</span><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'><o:p></o:p></span></p>
  </td>
  <td width=76 style='width:2.0cm;border:none;mso-border-left-alt:solid windowtext .5pt;
  padding:0cm .3pt 0cm .3pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>70.55<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:4;height:1.0pt'>
  <td width=104 style='width:78.0pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>Deeplabv3+<o:p></o:p></span></p>
  </td>
  <td width=76 style='width:2.0cm;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>39.76<o:p></o:p></span></p>
  </td>
  <td width=76 style='width:2.0cm;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>32<o:p></o:p></span></p>
  </td>
  <td width=66 style='width:49.6pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>43.30<o:p></o:p></span></p>
  </td>
  <td width=85 style='width:63.8pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-font-kerning:
  0pt;mso-fareast-language:EN-US'>77.31</span><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'><o:p></o:p></span></p>
  </td>
  <td width=76 style='width:2.0cm;border:none;mso-border-left-alt:solid windowtext .5pt;
  padding:0cm .3pt 0cm .3pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>74.70<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:5;height:1.0pt'>
  <td width=104 style='width:78.0pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>Swin-<span
  style='color:black;mso-themecolor:text1'>Un</span>et</span></span><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'><o:p></o:p></span></p>
  </td>
  <td width=76 style='width:2.0cm;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>41.42<o:p></o:p></span></p>
  </td>
  <td width=76 style='width:2.0cm;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>22<o:p></o:p></span></p>
  </td>
  <td width=66 style='width:49.6pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>0.02<o:p></o:p></span></p>
  </td>
  <td width=85 style='width:63.8pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>59.72<o:p></o:p></span></p>
  </td>
  <td width=76 style='width:2.0cm;border:none;mso-border-left-alt:solid windowtext .5pt;
  padding:0cm .3pt 0cm .3pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>57.19<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:6;height:1.0pt'>
  <td width=104 style='width:78.0pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>UNetformer</span></span><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'><o:p></o:p></span></p>
  </td>
  <td width=76 style='width:2.0cm;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>24.19<o:p></o:p></span></p>
  </td>
  <td width=76 style='width:2.0cm;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>23<o:p></o:p></span></p>
  </td>
  <td width=66 style='width:49.6pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>6.03<o:p></o:p></span></p>
  </td>
  <td width=85 style='width:63.8pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>77.73<o:p></o:p></span></p>
  </td>
  <td width=76 style='width:2.0cm;border:none;mso-border-left-alt:solid windowtext .5pt;
  padding:0cm .3pt 0cm .3pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>74.95<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:7;height:1.0pt'>
  <td width=104 style='width:78.0pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm .3pt 0cm .3pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>Segformer</span></span><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'><o:p></o:p></span></p>
  </td>
  <td width=76 style='width:2.0cm;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm .3pt 0cm .3pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>84.59<o:p></o:p></span></p>
  </td>
  <td width=76 style='width:2.0cm;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm .3pt 0cm .3pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>18<o:p></o:p></span></p>
  </td>
  <td width=66 style='width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm .3pt 0cm .3pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>11.65<o:p></o:p></span></p>
  </td>
  <td width=85 style='width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm .3pt 0cm .3pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>77.54<o:p></o:p></span></p>
  </td>
  <td width=76 style='width:2.0cm;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>75.23<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:8;mso-yfti-lastrow:yes;height:5.65pt'>
  <td width=104 style='width:78.0pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm .3pt 0cm .3pt;height:5.65pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=SpellE><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>SRCBTFusion</span></span><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>-Net
  <o:p></o:p></span></p>
  </td>
  <td width=76 style='width:2.0cm;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;height:5.65pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>84.94<o:p></o:p></span></p>
  </td>
  <td width=76 style='width:2.0cm;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;height:5.65pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>21<o:p></o:p></span></p>
  </td>
  <td width=66 style='width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;height:5.65pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>17.48<o:p></o:p></span></p>
  </td>
  <td width=85 style='width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;height:5.65pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-font-kerning:0pt;mso-fareast-language:EN-US'>78.12</span></b><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'><o:p></o:p></span></p>
  </td>
  <td width=76 style='width:2.0cm;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm .3pt 0cm .3pt;
  height:5.65pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-fareast-language:EN-US'>77.83<o:p></o:p></span></b></p>
  </td>
 </tr>
</table>

![image](https://github.com/js257/SRCBTFusion-Net/figure/fig10.jpg)<br>
Fig. 1. Examples of semantic segmentation results of different models on Potsdam dataset, the last column shows the predictions of our SRCBTFusion-Net, GT represents real label.<br>

![image](https://github.com/js257/SRCBTFusion-Net/figure/fig11.jpg)<br>
Fig. 2. Examples of semantic segmentation results of different models on Vaihingen dataset, the last column shows the predictions of our SRCBTFusion-Net, GT represents real label.<br>
# Requirement
<pre>Python 3.7.0+
Pytorch 1.2.0
CUDA 11.2
tqdm 4.65.0
</pre>
