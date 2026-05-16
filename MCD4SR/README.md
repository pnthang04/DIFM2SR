# MCD4SR: Multimodal Collaborative Denoising with Modality Balancing for Sequential Recommendation

## Introduction

### datasets


### train
CUDA_VISIBLE_DEVICES=0 nohup python train_denoiser_main.py --benchmark Amazon --dataset beauty --lr_encoder 1e-4 --temperature 0.2 --exp_name amazon_beauty_lrenc0001 > ./log/amazon_beauty_lrenc0001.log 2>&1 &


### Acknowledgements
Our code is based on the implementation of [TIGER](https://github.com/XiaoLongtaoo/TIGER).
  
