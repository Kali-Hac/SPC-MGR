![Python >=3.5](https://img.shields.io/badge/Python->=3.5-blue.svg)
![PyTorch >=1.6](https://img.shields.io/badge/Tensorflow->=1.10-yellow.svg)
# SPC-MGR
Skeleton Prototype Contrastive Learning with Multi-level Graph Relation Modeling for Unsupervised Person Re-Identification

## Introduction
This is the official implementation of SPC-MGR presented by "Skeleton Prototype Contrastive Learning with Multi-level Graph Relation Modeling for Unsupervised Person Re-Identification". The codes are used to reproduce experimental results in the [paper](./).

<!-- ![image](https://github.com/Kali-Hac/SM-SGE/blob/main/img/overview.png) -->

<!-- Abstract: Person re-identification via 3D skeletons is an emerging topic with great potential in security-critical applications. Existing methods typically learn body and motion features from the body-joint trajectory, whereas they lack a systematic way to model body structure and underlying relations of body components beyond the scale of body joints. In this paper, we for the first time propose a Self-supervised Multi-scale Skeleton Graph Encoding (SM-SGE) framework that comprehensively models human body, component relations, and skeleton dynamics from unlabeled skeleton graphs of various scales to learn an effective skeleton representation for person Re-ID. Specifically, we first devise multi-scale skeleton graphs with coarse-to-fine human body partitions, which enables us to model body structure and skeleton dynamics at multiple levels. Second, to mine inherent correlations between body components in skeletal motion, we propose a multi-scale graph relation network to learn structural relations between adjacent body-component nodes and collaborative relations among nodes of different scales, so as to capture more discriminative skeleton graph features. Last, we propose a novel multi-scale skeleton reconstruction mechanism to enable our framework to encode skeleton dynamics and high-level semantics from unlabeled skeleton graphs, which encourages learning a discriminative skeleton representation for person Re-ID. Extensive experiments show that SM-SGE outperforms most state-of-the-art skeleton-based methods. 
We further demonstrate its effectiveness on 3D skeleton data estimated from large-scale RGB videos. -->

## Requirements
- Python 3.5
- Tensorflow 1.10.0 (GPU)

## Datasets and Models
We provide three already pre-processed datasets (IAS-Lab, BIWI, KGBD) with various sequence lengths on <br/>
<!-- https://pan.baidu.com/s/1nuFH2EENyrMZbTnKGYssTw  &nbsp; &nbsp; &nbsp; password：&nbsp;  hyo7  <br/> -->

**Note**: The access to the Vislab Multi-view KS20 dataset and large-scale RGB-based gait dataset CASIA-B are available upon request. If you have signed the license agreement and been granted the right to use them, please email me with the signed agreement and I will share the complete pre-processed KS20 and CASIA-B data.

All the best models reported in our paper can be acquired on <br/> 
<!-- https://pan.baidu.com/s/1AIn7Iyfn7B-w2Eh3ZfHIZA &nbsp; &nbsp; &nbsp; password：&nbsp; sd4v  <br/>  -->
Please download the pre-processed datasets ``Datasets/`` and model files ``re-ID_Models/`` into the current directory. <br/>


The original datasets can be downloaded here: [IAS-Lab](http://robotics.dei.unipd.it/reid/index.php/downloads), [BIWI](http://robotics.dei.unipd.it/reid/index.php/downloads), [KGBD](https://www.researchgate.net/publication/275023745_Kinect_Gait_Biometry_Dataset_-_data_from_164_individuals_walking_in_front_of_a_X-Box_360_Kinect_Sensor), [KS20](http://vislab.isr.ist.utl.pt/datasets/#ks20), [CASIA-B](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp). We also provide the ``Preprocess.py`` for directly transforming original datasets to the formated training and testing data. <br/> 

## Dataset Pre-processing
To (1) extract 3D skeleton sequences of length **f** from original datasets and (2) process them in a unified format (``.npy``) for the model inputs, please simply run the following command: 
```bash
python Pre-process.py --dataset KGBD --f 6

```
**Note**: If you hope to preprocess manually (or *you can get the already preprocessed data [**here**](./)*), please frist download and unzip the original datasets to the current directory with following folder structure:
```bash
[Current Directory]
├─ BIWI
│    ├─ Testing
│    │    ├─ Still
│    │    └─ Walking
│    └─ Training
├─ IAS
│    ├─ TestingA
│    ├─ TestingB
│    └─ Training
├─ KGBD
│    └─ kinect gait raw dataset
└─ KS20
     ├─ frontal
     ├─ left_diagonal
     ├─ left_lateral
     ├─ right_diagonal
     └─ right_lateral
```
After dataset preprocessing, the auto-generated folder structure of datasets is as follows:
```bash
[Current Directory]
Datasets
├─ BIWI
│    └─ 6
│      ├─ test_npy_data
│      │    ├─ Still
│      │    └─ Walking
│      └─ train_npy_data
├─ IAS
│    └─ 6
│      ├─ test_npy_data
│      │    ├─ A
│      │    └─ B
│      └─ train_npy_data
├─ KGBD
│    └─ 6
│      ├─ test_npy_data
│      │    ├─ gallery
│      │    └─ probe
│      └─ train_npy_data
└─ KS20
    └─ 6
      ├─ test_npy_data
      │    ├─ gallery
      │    └─ probe
      └─ train_npy_data
```

## Model Usage

To (1) train the unsupervised SPC-MGR to obtain multi-level skeleton graph representations and (2) validate their effectiveness on the person re-ID task on a specific dataset (probe), please simply run the following command:  

```bash
python SPC-MGR.py --dataset KS20 --probe probe

# Default options: --dataset KS20 --probe probe --length 6  --gpu 0
# --dataset [IAS, KS20, BIWI, KGBD, CASIA_B]
# --probe ['probe' (the only probe for KS20 or KGBD), 'A' (for IAS-A probe), 'B' (for IAS-B probe), 'Walking' (for BIWI-Walking probe), 'Still' (for BIWI-Still probe)] 
# --length [4, 6, 8, 10, 12] 
# --(t, lr, eps, min_samples, m, fusion_lambda) with default settings for each dataset
# --mode [UF (for unsupervised training), DG (for direct domain generalization)]
# --gpu [0, 1, ...]

```
Please see ```SPC-MGR.py``` for more details.

To print evaluation results (Top-1, Top-5, Top-10 Accuracy, mAP) of the best model saved in default directory (```re-ID_Models/(Dataset)/(Probe)```), run:

```bash
python SPC-MGR.py --dataset KS20 --probe probe --evaluate 1
```

## Application to Model-Estimated Skeleton Data 
To apply our SPC-MGR to person re-ID under the large-scale RGB setting (CASIA B), we exploit pose estimation methods to extract 3D skeletons from RGB videos of CASIA B as follows:
- Step 1: Download [CASIA-B Dataset](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)
- Step 2: Extract the 2D human body joints by using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- Step 3: Estimate the 3D human body joints by using [3DHumanPose](https://github.com/flyawaychase/3DHumanPose)

We provide already pre-processed skeleton data of CASIA B for **single-condition** (Nm-Nm, Cl-Cl, Bg-Bg) and **cross-condition evaluation** (Cl-Nm, Bg-Nm) (f=40/50/60) on 
<!-- &nbsp; &nbsp; &nbsp; https://pan.baidu.com/s/1gDekBzf-3bBVdd0MGL0CvA &nbsp; &nbsp; &nbsp; password：&nbsp;  x3e5 <br/> -->
Please download the pre-processed datasets into the directory ``Datasets/``. <br/>

## Usage
To (1) train the SPC-MGR to obtain skeleton representations and (2) validate their effectiveness on the person re-ID task on CASIA B under **single-condition** and **cross-condition** settings, please simply run the following command:

```bash
python SPC-MGR.py --dataset CAISA_B --probe_type nm.nm --length 40

# --length [40, 50, 60] 
# --probe_type ['nm.nm' (for 'Nm' probe and 'Nm' gallery), 'cl.cl', 'bg.bg', 'cl.nm' (for 'Cl' probe and 'Nm' gallery), 'bg.nm']  
# --(t, lr, eps, min_samples, m, fusion_lambda) with default settings
# --gpu [0, 1, ...]

```

## Application to Generalized Person re-ID Task
To transfer the SPC-MGR model trained on a **source** dataset to a new **target** dataset and further **fine-tune** with the unlabeled target data for the generalized person re-ID task, please simply run the following command:  
```bash
python SPC-MGR.py --dataset KS20 --probe probe --S_dataset KGBD --S_probe probe --mode DG

# Target dataset: --dataset [IAS, KS20, BIWI, KGBD, CASIA_B]
# Source dataset: --S_dataset [IAS, KS20, BIWI, KGBD, CASIA_B]
# Target probe: --probe ['probe' (for KS20 or KGBD), 'A' (for IAS-A probe), 'B' (for IAS-B probe), 'Walking' (for BIWI-Walking probe), 'Still' (for BIWI-Still probe)] 
# Source probe: --S_probe ['probe' (for KS20 or KGBD), 'A' (for IAS-A probe), 'B' (for IAS-B probe), 'Walking' (for BIWI-Walking probe), 'Still' (for BIWI-Still probe)] 
# --(t, lr, eps, min_samples, m, fusion_lambda) with default settings for the source dataset
# --mode [DG (for direct domain generalizatio/transferring)]
# --gpu [0, 1, ...]

```

**Note**: This task requires first training a model on a specific probe set (```--S_probe```) of the source dataset (```--S_dataset```), and then applying this model to a specific target probe set (```--probe```) of the target dataset (```--dataset```). Our code will automatically fine-tune the pre-trained model on the unlabled data of target dataset.

Please see ```SPC-MGR.py``` for more details.


<!-- ## Results -->
<!-- ![results](img/SM-SGE-results.png) -->

# Acknowledgements

The SPC-MGR is built based in part on the project of [GAT](https://github.com/PetarV-/GAT) and part on our project of [MG-SCR](https://github.com/Kali-Hac/MG-SCR). Thanks to Veličković *et al.* for opening source of their excellent works [GAT](https://github.com/PetarV-/GAT). 

## License

SPC-MGR is released under the MIT License.
