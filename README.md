# ESAM++: Efficient Online 3D Perception on the Edge

This repo implements the core component, an efficient point cloud encoder, in the paper ```ESAM++: Efficient Online 3D Perception on the Edge```. 

## Overview
ESAM++ is an efficient online 3D perception model that achieves real-time performance using only CPU. This repo implements our core contribution: an efficient hirerarchical sparse feature pyramid network. 

More details can be found on the project page ([https://github.com/google/esamplusplus](https://github.com/google/esamplusplus)).

## Getting Started
Please follow [ESAM](https://github.com/xuxw98/ESAM) (ICLR 2025) for environment setup, dataset preparation, and training and evaluation. To create a new conda environment and activate it:
```
# we recommend python 3.12, though ESAM uses 3.8.
conda create -n esamplusplus python=3.12
conda activate esamplusplus
```

## Training & Evaluation
Note that this codebase does not provide full code for training and evaluation. To train and evaluate our method, the users need to first install [ESAM](https://github.com/xuxw98/ESAM) and then use our backbone (```src/backbone.py```) and confirguation file (```configs/esamplusplus_online_scannet200_CA.py```). Please contact the first author (qinliu2020@gmail.com) if you experienced any difficulties.


## Benchmarks
We benchmark our method using the following datasets: [ScanNet](https://github.com/ScanNet/ScanNet), [SceneNN](https://github.com/hkust-vgd/scenenn), and [3RScan](https://github.com/WaldJohannaU/3RScan). We follow the same training and evaluation settings proposed in [ESAM](https://github.com/xuxw98/ESAM).

## Acknowledgement
Our codebase is based on [ESAM](https://github.com/xuxw98/ESAM). We thank the authors for the great work!

## Contributors
- **Qin Liu (Stanford University)**
- **Lavisha Aggarwal (Google)**
- **Vikas Bahirwani (Google)**
- **Lin Li (Google)**
- **Aleksander Holynski (Google)**
- **Saptarashmi Bandyopadhyay (Google)**
- **Zhengyang Shen (Google)**
- **Marc Niethammer (UCSD)**
- **Ehsan Adeli (Stanford University)**
- **Andrea Colaco (Google)**
