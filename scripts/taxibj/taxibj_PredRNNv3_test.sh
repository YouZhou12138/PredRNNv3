cd ..
cd ..
python test.py \
    --setting  configs/taxibj/PredRNNv3/6M/seed=27.yaml \
    --checkpoint experiments/taxibj/PredRNNv3/6M/checkpoints/epoch=128-MSE=0.2795.ckpt \
    --data_name  taxibj \
    --data_dir D:\dataset \
    --pred_dir experiments/taxibj/PredRNNv3/6M/predictions
