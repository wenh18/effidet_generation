python train.py  /mscoco --model resdet50 -b 16 --amp  --lr .09 --warmup-epochs 5  --sync-bn --opt fusedmomentum  --dataset coco2017 # --model-ema