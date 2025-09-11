# PredRNNv3: A CNN-Transformer Collaborative Recurrent Neural Network for Spatiotemporal Prediction
The manuscript is currently under peer review by the journal Neurocomputing (as of September 11, 2025). To facilitate reviewers' full appreciation of the completeness of the study implementation, we hereby provide the accompanying code repository. The provided code can directly reproduce all experimental results reported in the paper. We sincerely hope that this will serve as a useful reference for the review process.
## Get Started
1.Install the PyTorch-related tools as instructed in the requirement.txt file.

2.According to the instructions in the "datasets/prepare-data" folder, download the training data sets required, and store the downloaded data sets in the designated location.

3.Train the model. You can use the following bash script to train the model. The learned model will be saved in the 'experiments/data_name/Model_version/checkpoints' folder. At the same time, you can utilize the pre-trained model weights and evaluate the prediction performance of the model by following the bash script provided below. The generated future frames will be saved in the --pred_dir folder. 
cd scripts/mmnist/
sh mm_PredRNNv3_train.sh
sh mm_PredRNNv3_test.sh.sh

4.You can get pretrained models from https://pan.baidu.com/s/1qc3v2yA5djtz2VthMGHW4w?pwd=cqtb After downloading, please extract the compressed file to the "PredRNNv3" folder.


