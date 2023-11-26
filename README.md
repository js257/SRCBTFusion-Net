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
<table class="MsoTableGrid" border="1" cellspacing="0" cellpadding="0" width="469" style="width:351.8pt;border-collapse:collapse;border:none;mso-border-alt:solid windowtext .5pt;
 mso-yfti-tbllook:1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes;height:6.95pt">
  <td width="123" rowspan="2" style="width:92.15pt;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:double windowtext 1.5pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:double windowtext 1.5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">Method<o:p></o:p></span></p>
  </td>
  <td width="85" rowspan="2" style="width:63.8pt;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:double windowtext 1.5pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  mso-border-bottom-alt:double windowtext 1.5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">Parameters (M)<o:p></o:p></span></p>
  </td>
  <td width="76" rowspan="2" style="width:2.0cm;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:double windowtext 1.5pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  mso-border-bottom-alt:double windowtext 1.5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">Speed (FPS)<o:p></o:p></span></p>
  </td>
  <td width="57" rowspan="2" style="width:42.55pt;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:double windowtext 1.5pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  mso-border-bottom-alt:double windowtext 1.5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">Flops (G)<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  mso-border-left-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体;color:black;mso-themecolor:text1">Potsdam</span><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体"><o:p></o:p></span></p>
  </td>
  <td width="63" style="width:47.0pt;border:solid windowtext 1.0pt;border-right:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">Vaihingen<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:1;height:6.95pt">
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  double windowtext 1.5pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;mso-border-bottom-alt:double windowtext 1.5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">MIoU (%)<o:p></o:p></span></p>
  </td>
  <td width="63" style="width:47.0pt;border:none;border-bottom:double windowtext 1.5pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">MIoU (%)<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:2;height:6.95pt">
  <td width="123" style="width:92.15pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:double windowtext 1.5pt;mso-border-top-alt:double windowtext 1.5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span class="SpellE"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;
  mso-fareast-font-family:宋体">TransUNet</span></span><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体"><o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:double windowtext 1.5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:double windowtext 1.5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">76.77<o:p></o:p></span></p>
  </td>
  <td width="76" style="width:2.0cm;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:double windowtext 1.5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:double windowtext 1.5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">19<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:double windowtext 1.5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:double windowtext 1.5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">15.51<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:double windowtext 1.5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:double windowtext 1.5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">76.86<o:p></o:p></span></p>
  </td>
  <td width="63" style="width:47.0pt;border:none;mso-border-top-alt:double windowtext 1.5pt;
  mso-border-left-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">74.30<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:3;height:6.95pt">
  <td width="123" style="width:92.15pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span class="SpellE"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;
  mso-fareast-font-family:宋体">ABCNet</span></span><!--[if supportFields]><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体'><span style='mso-element:field-begin'></span> REF
  _Ref135246882 \r \h<span style='mso-spacerun:yes'>&nbsp; </span>\*
  MERGEFORMAT <span style='mso-element:field-separator'></span></span><![endif]--><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;
  mso-fareast-font-family:宋体"><span style="mso-spacerun:yes">&nbsp;</span><!--[if gte mso 9]><xml>
   <w:data>08D0C9EA79F9BACE118C8200AA004BA90B02000000080000000E0000005F005200650066003100330035003200340036003800380032000000</w:data>
  </xml><![endif]--></span><!--[if supportFields]><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体'><span style='mso-element:field-end'></span></span><![endif]--><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;
  mso-fareast-font-family:宋体"><o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">28.57<o:p></o:p></span></p>
  </td>
  <td width="76" style="width:2.0cm;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">26<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">7.24<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">74.89<o:p></o:p></span></p>
  </td>
  <td width="63" style="width:47.0pt;border:none;mso-border-left-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">70.55<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:4;height:6.95pt">
  <td width="123" style="width:92.15pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">Deeplabv3+ <o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">39.76<o:p></o:p></span></p>
  </td>
  <td width="76" style="width:2.0cm;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">32<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">43.30<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">77.31<o:p></o:p></span></p>
  </td>
  <td width="63" style="width:47.0pt;border:none;mso-border-left-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">74.70<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:5;height:6.95pt">
  <td width="123" style="width:92.15pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">Swin-<span class="SpellE"><span style="color:black;mso-themecolor:text1">Un</span>et</span><o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">41.42<o:p></o:p></span></p>
  </td>
  <td width="76" style="width:2.0cm;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">22<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">0.02<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">59.72<o:p></o:p></span></p>
  </td>
  <td width="63" style="width:47.0pt;border:none;mso-border-left-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">57.19<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:6;height:6.95pt">
  <td width="123" style="width:92.15pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span class="SpellE"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;
  mso-fareast-font-family:宋体">UNetformer</span></span><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体;color:#0070C0"> </span><span lang="EN-US" style="font-size:8.0pt;font-family:
  &quot;Times New Roman&quot;,serif;mso-fareast-font-family:宋体"><o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">24.19<o:p></o:p></span></p>
  </td>
  <td width="76" style="width:2.0cm;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">23<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">6.03<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">77.90<o:p></o:p></span></p>
  </td>
  <td width="63" style="width:47.0pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">74.95<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:7;height:6.95pt">
  <td width="123" style="width:92.15pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span class="SpellE"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;
  mso-fareast-font-family:宋体">Segformer</span></span><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体"> <o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">84.59<o:p></o:p></span></p>
  </td>
  <td width="76" style="width:2.0cm;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">18<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">11.65<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">77.70<o:p></o:p></span></p>
  </td>
  <td width="63" style="width:47.0pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">75.23<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:8;mso-yfti-lastrow:yes;height:6.95pt">
  <td width="123" style="width:92.15pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span class="SpellE"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;
  mso-fareast-font-family:宋体">SRCBTFusion</span></span><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">-Net<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">86.30<o:p></o:p></span></p>
  </td>
  <td width="76" style="width:2.0cm;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">28<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;mso-fareast-font-family:
  宋体">22.58<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><b><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif">78.62</span></b><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;
  mso-fareast-font-family:宋体"><o:p></o:p></span></p>
  </td>
  <td width="63" style="width:47.0pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:6.95pt">
  <p class="MsoNormal" align="center" style="text-align:center"><b><span lang="EN-US" style="font-size:8.0pt;font-family:&quot;Times New Roman&quot;,serif;
  mso-fareast-font-family:宋体">76.27<o:p></o:p></span></b></p>
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
  volume={},
  number={},
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
