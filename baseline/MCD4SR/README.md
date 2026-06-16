# MCD4SR: Multimodal Collaborative Denoising with Modality Balancing for Sequential Recommendation

## Introduction

### datasets


### train
CUDA_VISIBLE_DEVICES=0 nohup python train_denoiser_main.py --benchmark Amazon --dataset beauty --lr_encoder 1e-4 --temperature 0.2 --exp_name amazon_beauty_lrenc0001 > ./log/amazon_beauty_lrenc0001.log 2>&1 &

### Baby Modern SigLIP Large
Download and prepare `thangkt/baby-modern-bge-siglip` for MCD4SR:

```bash
python scripts/prepare_baby_modern_mcd.py
```

Train and evaluate with MCD4SR:

```bash
mkdir -p ./log
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python train_denoiser_main.py --benchmark Amazon --dataset baby_modern --lr_encoder 1e-4 --temperature 0.2 --exp_name baby_modern_siglip_large_bs1024
```

Config used:

```yaml
train_batch_size : 1024
test_batch_size : 512
```

#### Baby Modern SigLIP Large Results

Training stopped early at epoch 54 after 10 consecutive validation evaluations without improving validation NDCG@20.

Best model is selected by validation NDCG@20 and saved at:

```text
result/experiments/Amazon/baby_modern_siglip_large_bs1024/best_model.pt
```

Best validation epoch: 44

| Metric | @5 | @10 | @20 | @50 |
| --- | ---: | ---: | ---: | ---: |
| Recall | 0.0410 | 0.0653 | 0.0951 | 0.1560 |
| NDCG | 0.0249 | 0.0327 | 0.0401 | 0.0521 |
| Precision | 0.0082 | 0.0065 | 0.0048 | 0.0031 |
| MRR | 0.0196 | 0.0228 | 0.0248 | 0.0267 |

Best test epoch: 50

| Metric | @5 | @10 | @20 | @50 |
| --- | ---: | ---: | ---: | ---: |
| Recall | 0.0308 | 0.0483 | 0.0720 | 0.1194 |
| NDCG | 0.0202 | 0.0259 | 0.0318 | 0.0412 |
| Precision | 0.0062 | 0.0048 | 0.0036 | 0.0024 |
| MRR | 0.0168 | 0.0191 | 0.0207 | 0.0222 |


### Acknowledgements
Our code is based on the implementation of [TIGER](https://github.com/XiaoLongtaoo/TIGER).
  
