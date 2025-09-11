cd ..
cd ..
python test.py \
    --setting  configs\moving_mnist\PredRNNv3\6M\seed=1234.yaml \
    --checkpoint experiments\moving_mnist\PredRNNv3\6M\checkpoints\epoch=1529-MSE=12.4183.ckpt\
    --data_name  moving_mnist \
    --data_dir D:\dataset \
    --pred_dir experiments/moving_mnist/PredRNNv3/6M/predictions
