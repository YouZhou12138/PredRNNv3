# PredRNNv3: A CNN-Transformer Collaborative Recurrent Neural Network for Spatiotemporal Prediction
This repository is an open-source project implementing video prediction benchmarks using the **PredRNNv3** model, which includes the implementation code corresponding to the manuscript. Currently, this project is intended solely for reviewers' reference.
##  Introduction
In this work, we replace the spatiotemporal LSTM employed in **[PredRNNv2](https://github.com/thuml/predrnn-pytorch?tab=readme-ov-file)** with a Dual-Branch LSTM (DB-LSTM). Specifically, the CNN branch is responsible for local feature extraction, while the ViT branch handles global feature extraction. The Dual-Branch Modulation Module (DBMM) is designed to efficiently fuse the local and global features.

The main framework of DB-LSTM.
<img src="https://github.com/YouZhou12138/PredRNNv3/blob/main/imgs/DB_LSTM.png" style="display: block; margin: 0 auto; width: 50%;" />
![](https://github.com/YouZhou12138/PredRNNv3/blob/main/imgs/DB_LSTM.png)
Quantitative results of different methods on the Moving MNIST dataset (10→10 frames).
![](https://github.com/YouZhou12138/PredRNNv3/blob/main/imgs/Moving_mnist.png)
## Overview
- ```PredRNNv3/configs```: Including the hyperparameter settings tested on all benchmark datasets.
- ```PredRNNv3/datasets```: Including datasets acquisition and preprocessing methods used in the manuscript.
- ```PredRNNv3/experiments```: This repository contains the training trial logs and model weights achieving optimal performance metrics. Note that these components are not hosted on GitHub. To reproduce the results presented in the manuscript, you may retrieve our pre-trained model weights from [Baidu Pan](https://pan.baidu.com/s/1qc3v2yA5djtz2VthMGHW4w?pwd=cqtb) and place the `experiments`folder inside the `PredRNNv3`directory.
- ```PredRNNv3/models```: It includes the main network architecture of PredRNNv3 and also provides the implementation of the PredFormer model (for reference only).
- ```PredRNNv3/train```: This is an executable Python script designed for the training of all benchmark models and data.

## Installation
Please refer to the `requirements.txt`file for the required Python libraries necessary for the code implementation.

## Getting Started
You can use the following bash script to train the model. The learned model will be saved in the `experiments/data_name/Model_version/checkpoints` folder. At the same time, you can utilize the pre-trained model weights and evaluate the prediction performance of the model by following the bash script provided below. The generated future frames will be saved in the `--pred_dir` folder.
```pythonscript
cd scripts/mmnist/
sh mm_PredRNNv3_train.sh
sh mm_PredRNNv3_test.sh
```
Alternatively, one can directly pass parameters to perform model training and testing.
For train:
```pythonscript
 python train.py --setting configs/moving_mnist/PredRNNv3/6M/seed=1234.yaml --data_name moving_mnist --data_dir D:\dataset
```
For test:
```pythonscript
 python test.py --setting configs/moving_mnist/PredRNNv3/6M/seed=1234.yaml --checkpoint experiments\moving_mnist\PredRNNv3\6M\checkpoints\epoch=1529-MSE=12.4183.ckpt --data_name moving_mnist --data_dir D:\dataset --pred_dir experiments/moving_mnist/PredRNNv3/6M/predictions
```


