cd ..
cd ..
python test.py \
    --setting  configs/weather/PredRNNv3/6M/seed=27.yaml \
    --checkpoint experiments/weather/PredRNNv3/6M/checkpoints/epoch=26-MSE=1.3371.ckpt \
    --data_name  weather \
    --data_dir D:\dataset \
    --pred_dir experiments/weather/PredRNNv3/6M/predictions
