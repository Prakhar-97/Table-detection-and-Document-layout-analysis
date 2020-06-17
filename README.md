# Table-detection-and-Document-layout-analysis
## Introduction
Using State of the Art techniques for table detection and Document layout analysis. For table detection we are using MMDetection version(1.2), however in Document layout analysis we are using the models which have been developed in MMDetection version(2.0)
 
## Setup
<b>Models are developed in Pytorch based <a href="https://github.com/open-mmlab/mmdetection">MMdetection</a> framework (Version 2.0)</b>
<br>

<pre>
git clone -'https://github.com/open-mmlab/mmdetection.git'
cd "mmdetection"
python setup.py install
python setup.py develop
pip install -r {"requirements.txt"}
</pre>

## Image Augmentation
We have followed Dilation and Smudge techniques for Data Augmentation

<img src="Data Preparation/Images/3img.png" width="750"/><br>


## Model Zoo
Config file for the Models :


1. For table detection
<a href="CascadeTab/Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py/">Config_file</a><br>

2. For Document Analysis
<a href="Document layout analysis/Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py">Config_file</a><br>

Note: Config paths are only required to change during training

Checkpoints of the Models that have been trained : 

<table>
  <tr>
  <th>Model Name</th><th>Checkpoint File</th>
  </tr>
  <tr>
  <td>Table structure recognition</td><td><a href="https://drive.google.com/open?id=1-vjfGRhF8kqvKwZPPFNwiTaOoonJlGgv">Checkpoint</a></td>
  </tr>
  <tr>
  <td>Document layout analysis</td><td><a href="https://drive.google.com/file/d/1TGMMdk9WDY_xOqb3IrD0G1DzncMiAP0T/view?usp=sharing">Checkpoint</a></td>
  </tr>
</table>

## Datasets
1. Table detection and Structure Recignition:
You can refer to <a href="https://github.com/DevashishPrasad/CascadeTabNet">Dataset</a> to have a better understanding of the Dataset

2. Document layout Analysis:
You can refer to <a href="https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/PubLayNet.html">Dataset</a> to have a better understanding of the dataset.

## Training

Refer to the two colab notebooks thathave been mentioned as they will direct you through the steps that need to be followed. If using a custom dataset do go through <a href="https://mmdetection.readthedocs.io/en/latest/">MMdet Docs</a>


