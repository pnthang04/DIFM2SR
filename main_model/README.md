# SGP4SR (AAAI'26)

Sequential recommendation model incorporating semantic and graph pooling.

## Project Structure

```
├── data/                    # Data loading and transformation code
│   ├── dataloader.py        # Custom dataloader with transforms
│   ├── dataset.py           # SGPDataset - loads embeddings and interactions
│   └── transform.py         # Data augmentation transforms
├── dataset/                 # Dataset archive and unzipped training files
│   ├── baby_modern_bge_siglip.tar.gz
│   └── baby_modern_raw_unzip/
├── run.py                   # Main training script
├── sgp.py                   # SGP model implementation
├── model_utils.py           # Model utilities
├── SGP4SR.yaml              # Model config
├── run.yaml                 # Training config
└── README.md
```

## Data

Data của project được unzip trực tiếp từ file:

```text
/kaggle/SGP4SR/SGP4SR/dataset/baby_modern_bge_siglip.tar.gz
```

File nén này đã chứa sẵn:

- `baby.train.inter`: tập train.
- `baby.valid.inter`: tập validation dùng trong quá trình train/early stopping.
- `baby.test.inter`: tập test dùng để đánh giá cuối cùng.
- `text_features_bge.npy`: vector text BGE, shape `(7015, 768)`.
- `image_features_siglip.npy`: vector image SigLIP, shape `(7015, 768)`.

### 1. Unzip Data

Chạy các lệnh sau từ thư mục project:

```bash
cd /kaggle/SGP4SR/SGP4SR

mkdir -p dataset/baby_modern_raw_unzip
tar -xzf dataset/baby_modern_bge_siglip.tar.gz -C dataset/baby_modern_raw_unzip
```

Sau khi unzip, folder data gốc sẽ có dạng:

```text
dataset/baby_modern_raw_unzip/
`-- baby_modern/
    |-- dataset/baby/
    |   |-- baby.train.inter
    |   |-- baby.valid.inter
    |   `-- baby.test.inter
    |-- text_features_bge.npy
    |-- image_features_siglip.npy
    |-- item_text.jsonl
    |-- image_paths.jsonl
    |-- image_download_failed.jsonl
    |-- item2id.json
    `-- user2id.json
```

### 2. Create RecBole File Links

RecBole đọc dataset theo tên folder và tên file cùng prefix. Vì folder train sẽ là `baby_modern_raw_unzip`, cần tạo các symlink sau:

```bash
cd /kaggle/SGP4SR/SGP4SR

ln -sf baby_modern/dataset/baby/baby.train.inter dataset/baby_modern_raw_unzip/baby_modern_raw_unzip.train.inter
ln -sf baby_modern/dataset/baby/baby.valid.inter dataset/baby_modern_raw_unzip/baby_modern_raw_unzip.valid.inter
ln -sf baby_modern/dataset/baby/baby.test.inter dataset/baby_modern_raw_unzip/baby_modern_raw_unzip.test.inter
ln -sf baby_modern/text_features_bge.npy dataset/baby_modern_raw_unzip/baby_modern_raw_unzip.text
ln -sf baby_modern/image_features_siglip.npy dataset/baby_modern_raw_unzip/baby_modern_raw_unzip.image
```

Sau bước này, folder dùng để train sẽ có các file/link chính:

```text
dataset/baby_modern_raw_unzip/
|-- baby_modern_raw_unzip.train.inter -> baby_modern/dataset/baby/baby.train.inter
|-- baby_modern_raw_unzip.valid.inter -> baby_modern/dataset/baby/baby.valid.inter
|-- baby_modern_raw_unzip.test.inter  -> baby_modern/dataset/baby/baby.test.inter
|-- baby_modern_raw_unzip.text        -> baby_modern/text_features_bge.npy
|-- baby_modern_raw_unzip.image       -> baby_modern/image_features_siglip.npy
`-- baby_modern/
```

### 3. Quick Check

Kiểm tra data/link đã có đúng chưa:

```bash
ls -l dataset/baby_modern_raw_unzip
```

Tên dataset dùng cho training là:

```text
baby_modern_raw_unzip
```

### All-in-One Setup

Có thể chạy toàn bộ bước unzip + tạo symlink bằng block sau:

```bash
cd /kaggle/SGP4SR/SGP4SR

mkdir -p dataset/baby_modern_raw_unzip
tar -xzf dataset/baby_modern_bge_siglip.tar.gz -C dataset/baby_modern_raw_unzip

ln -sf baby_modern/dataset/baby/baby.train.inter dataset/baby_modern_raw_unzip/baby_modern_raw_unzip.train.inter
ln -sf baby_modern/dataset/baby/baby.valid.inter dataset/baby_modern_raw_unzip/baby_modern_raw_unzip.valid.inter
ln -sf baby_modern/dataset/baby/baby.test.inter dataset/baby_modern_raw_unzip/baby_modern_raw_unzip.test.inter
ln -sf baby_modern/text_features_bge.npy dataset/baby_modern_raw_unzip/baby_modern_raw_unzip.text
ln -sf baby_modern/image_features_siglip.npy dataset/baby_modern_raw_unzip/baby_modern_raw_unzip.image
```

## Training

### Smoke Train

Chạy thử vài step để kiểm tra data, embedding, model forward/backward đều hoạt động:

```bash
cd /kaggle/SGP4SR/SGP4SR
python run.py -d baby_modern_raw_unzip --smoke-steps 2
```

Nếu chạy đúng, output sẽ có dạng:

```text
smoke step 1/2 loss=...
smoke step 2/2 loss=...
```

### Full Train

Chạy training full trên tập train, validate trên tập valid, và evaluate trên tập test:

```bash
cd /kaggle/SGP4SR/SGP4SR
python run.py -d baby_modern_raw_unzip
```

Kết quả evaluation được ghi theo logic hiện tại trong `run.py`, bao gồm `best_valid_result` và `test_result`.

Kết quả JSON được lưu trong thư mục:

```text
results/
```

## Experiment Controls

Các tham số chính đang được lưu trong:

- `SGP4SR.yaml`: model, fusion, graph pooling, BALW/SIMW, diffusion-denoiser.
- `run.yaml`: data path, batch size, learning rate, early stopping, epochs, metrics.

Config mặc định hiện tại:

```yaml
learning_rate: 0.0003
weight_decay: 1e-5
MAX_ITEM_LIST_LENGTH: 123
epochs: 50
stopping_step: 3
valid_metric: NDCG@20
hidden_dropout_prob: 0.4
attn_dropout_prob: 0.4
fusion_prior: [0.375, 0.5625, 0.0625]
fusion_dynamic_ratio: 0.1
w_balw: 0.001
balw_type: batch
balw_max_weight: 0.75
w_simw: 0.0
simw_mu: 2.0
simw_target: router
```

### SIMW Module

SIMW (`Semantic/Score-aware Ideal Modality Weight`) là regularization phụ cho router chọn trọng số giữa 3 nhánh `id/text/visual`.

Luồng tính trong `sgp.py`:

1. Model tính riêng logits cho `id`, `text`, `visual`.
2. Với item positive trong batch, `eva_imp` ước lượng độ khó/rank của từng modality.
3. `SIMW` chuyển độ khó thành `ideal_weights` bằng `softmax(-distance)`.
4. Loss KL-divergence kéo router weights về `ideal_weights`.

Các tham số:

- `w_simw`: hệ số loss SIMW. `0.0` nghĩa là tắt module.
- `simw_mu`: độ nhạy của ranking-feedback distance.
- `simw_target`: chọn weights để supervise, gồm `router` hoặc `final`.
- `simw_log_batches`: số batch đầu được in log `[simw]`.

Ví dụ bật SIMW khi chạy thử:

```bash
cd /kaggle/SGP4SR/SGP4SR
python run.py -d baby_modern_raw_unzip --smoke-steps 2 --w-simw 0.001 --simw-target router
```

Ví dụ chạy full experiment với SIMW:

```bash
cd /kaggle/SGP4SR/SGP4SR
python run.py -d baby_modern_raw_unzip -note simw_ --w-simw 0.001 --simw-mu 2.0 --simw-target router
```

### BALW Module

BALW là entropy-balance regularization cho fusion weights giữa `id/text/visual`.

Các tham số:

- `w_balw`: hệ số loss BALW.
- `balw_type`: `sample`, `batch`, hoặc `threshold`.
- `balw_max_weight`: ngưỡng max-weight khi dùng `threshold`.
- `balw_log_batches`: số batch đầu được in log fusion weights.

Ví dụ override BALW:

```bash
cd /kaggle/SGP4SR/SGP4SR
python run.py -d baby_modern_raw_unzip --w-balw 0.001 --balw-type batch --balw-max-weight 0.75
```

## Best Run Result

Best run hiện tại là run 2 với config:

```yaml
learning_rate: 0.0003
weight_decay: 1e-5
MAX_ITEM_LIST_LENGTH: 123
epochs: 10
stopping_step: 10
hidden_dropout_prob: 0.4
attn_dropout_prob: 0.4
```

Best validation ở epoch 8 theo `Recall@5`.

Best valid result:

```json
{
  "recall@5": 0.0443,
  "recall@10": 0.0706,
  "recall@20": 0.1022,
  "recall@50": 0.1675,
  "ndcg@5": 0.0271,
  "ndcg@10": 0.0355,
  "ndcg@20": 0.0435,
  "ndcg@50": 0.0564
}
```

Test result sau khi load best model:

```json
{
  "recall@5": 0.0355,
  "recall@10": 0.0576,
  "recall@20": 0.0848,
  "recall@50": 0.1404,  
  "ndcg@5": 0.0229,
  "ndcg@10": 0.0301,
  "ndcg@20": 0.0369,
  "ndcg@50": 0.0479
}
```

Best checkpoint:

```text
saved/SGP-May-09-2026_15-17-27.pth
```

Log chi tiết:

```text
log/SGP/SGP-baby_modern_raw_unzip-May-09-2026_15-16-43-9d0c6c.log
```

JSON kết quả:

```text
results/SGP-baby_modern_raw_unzip-2026-05-09_16-08-49.json
```

Config files:
- `SGP4SR.yaml` – Model hyperparameters
- `run.yaml` – Data & training configs
