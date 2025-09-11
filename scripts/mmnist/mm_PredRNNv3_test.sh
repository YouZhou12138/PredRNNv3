cd ..
cd ..
python test.py \
    --setting  configs\kth\PredRNNv3\6M\seed=27.yaml \
    --checkpoint experiments\kth\PredRNNv3\7M\checkpoints\epoch=42-PSNR=30.9075.ckpt\
    --data_name  kth \
    --data_dir D:\dataset \
    --pred_dir experiments\kth\PredRNNv3\7M\20
