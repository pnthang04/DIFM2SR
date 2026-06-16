# DIFM2SR: Denoised Intent-aware Future-aware Multimodal Sequential Recommendation

[Phạm Ngọc Thắng](mailto:pnthangkthl@gmail.com)<sup>✉</sup>  

![Task](https://img.shields.io/badge/Task-Multi--Modal-red)
![Task](https://img.shields.io/badge/Task-Recommendation-red)
![Framework](https://img.shields.io/badge/Framework-RecBole-blue)
![Model](https://img.shields.io/badge/Model-Sequential%20RecSys-green)

Quick Links:
[📦 Dataset](https://huggingface.co/datasets/thangkt/baby-modern-bge-siglip) |
[⚙️ Experiments](#chạy-thử-nghiệm) |
[📊 Metrics](#kết-quả) |
[🖼️ Architecture](#kiến-trúc)

## Giới Thiệu

Mô hình gợi ý tuần tự đa phương thức dùng lịch sử tương tác của người dùng cùng ba loại tín hiệu item: `ID`, `text`, `image`.

Mục tiêu chính:

- Giảm nhiễu trong text/image.
- Học tương tác giữa các modality.
- Fusion động để tránh một modality lấn át các modality khác.
- Tận dụng tín hiệu item tương lai trong training.

## Kiến Trúc

![Overall Architecture](img/fig_overall_architecture.png)

Pipeline gồm 4 phần:

1. `Denoised Representation Learning`: học riêng ID, text, image và giảm nhiễu từng modality.
2. `Cross-Modal Attentive MoE`: cho các modality trao đổi thông tin bằng Mixture of Experts.
3. `Intent-Aware Late Fusion`: mỗi modality dự đoán riêng, sau đó fusion bằng trọng số động.
4. `Future-Aware Auxiliary Learning`: dùng item tương lai trong training, không dùng khi inference.

### Module 1

![Module 1](img/fig_module1.png)

Học representation riêng cho từng modality, kết hợp denoising, graph co-occurrence và interest centers.

### Module 2

![Cross-Modal MoE](img/fig_cross_modal_moe.png)

MoE học quan hệ bổ trợ giữa `ID`, `text`, `image`, sau đó cập nhật từng modality theo residual.

### Module 3

![Module 3](img/fig_module3.png)

Fusion ở tầng logits:

```text
y = w_id * y_id + w_text * y_text + w_image * y_image
```

Các loss cân bằng giúp tránh weight collapse vào một modality duy nhất.

### Module 4

![Future Learning](img/fig_future_learning.png)

Future-aware learning dùng các item sau bước kế tiếp để học xu hướng dài hơn trong training.

## Dataset

Dataset Baby Modern BGE + SigLIP:

https://huggingface.co/datasets/thangkt/baby-modern-bge-siglip

Dữ liệu gồm:

- `train/valid/test interactions`
- text feature BGE
- image feature SigLIP
- mapping user/item

## Cài Đặt

```bash
pip install -r requirements.txt
```

## Chạy Thử Nghiệm

Kiểm tra nhanh:

```bash
cd difm2sr
python run.py -d baby_modern_raw_unzip --smoke-steps 2
```

Train đầy đủ:

```bash
cd difm2sr
python run.py -d baby_modern_raw_unzip
```

Kết quả được lưu trong:

```text
difm2sr/results/
```

## Metrics

Sử dụng các metric ranking:

- `Recall@K`
- `NDCG@K`

Config mặc định:

```yaml
topk: [5, 10, 20, 50]
metrics: [Recall, NDCG]
valid_metric: NDCG@20
```

## Kết Quả

Kết quả chính từ báo cáo:

| Dataset | Method | R@10 | R@20 | N@10 | N@20 |
| --- | --- | ---: | ---: | ---: | ---: |
| Baby | Proposed | **0.0596** | **0.0848** | **0.0430** | **0.0400** |
| Sports | Proposed | **0.0611** | **0.0902** | 0.0313 | **0.0388** |


## Cấu Trúc

```text
difm2sr/
  difm2sr.py        model DIFM2SR
  DIFM2SR.yaml      model config
  run.py            train/eval entrypoint
  run.yaml          data/eval config
baseline/      các baseline so sánh
img/           ảnh kiến trúc và module
pdf/           báo cáo và slide
requirements.txt
README.md
```
## Contact


Phạm Ngọc Thắng   
Email: pnthangkthl@gmail.com  
GitHub: https://github.com/pnthang04
