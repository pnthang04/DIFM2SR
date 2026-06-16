# DMMD4SR: Diffusion Model-based Multi-level Multimodal Denoising for Sequential Recommendation

This repository contains the code for "DMMD4SR: Diffusion Model-based Multi-level Multimodal Denoising for Sequential Recommendation" (**ACMMM 2025**).

## Model Architecture

The overall framework of our proposed DMMD4SR model is illustrated below:

![Model Framework](model_framework.png)

## Environment Setup

The code requires the following main dependencies:

*   Python == 3.9
*   PyTorch == 2.1.1

*Note: Other required packages are common libraries and can be installed using pip as needed.*

## Datasets

We evaluate our model on the following 5-core subsets of the Amazon Review Data:

*   Home & Kitchen (`Home`)
*   Beauty
*   Tools & Home Improvement (`Tools`)
*   Toys & Games (`Toys`)

The datasets can be found at: [UCSD Amazon Review Data](https://jmcauley.ucsd.edu/data/amazon/)

### Baby Modern BGE/SigLIP

This workspace also supports the Baby Modern dataset from Hugging Face:

```text
thangkt/baby-modern-bge-siglip
```

The prepared DMMD4SR links live in:

```text
process_data/baby_modern_raw_unzip.*
```

The text and image features are BGE/SigLIP embeddings with shape `(7015, 768)`, so run with `--pretrain_emb_dim 768`.


## How to Run

To run the experiment on the Beauty dataset, execute the following script:

```bash
bash run_beauty.sh
```

*(You may need to adapt the script or create similar ones for other datasets.)*

To run the Baby Modern BGE/SigLIP experiment:

```bash
cd /kaggle/SGP4SR/DMMD4SR/src

python -u main.py \
  --data_dir ../process_data/ \
  --data_name baby_modern_raw_unzip \
  --data_format recbole \
  --text_embedding_path ../process_data/baby_modern_raw_unzip.text \
  --image_embedding_path ../process_data/baby_modern_raw_unzip.image \
  --pretrain_emb_dim 768 \
  --output_dir output_baby_modern_full/ \
  --epochs 80 \
  --batch_size 256 \
  --max_seq_length 50 \
  --model_idx 0
```

## Baby Modern Result

Run:

```text
ICLRec-SAS-baby_modern_raw_unzip-0
```

Result:

```json
{
  "recall@5": 0.0321,
  "recall@10": 0.0486,
  "recall@20": 0.0699,
  "recall@50": 0.1199,
  "ndcg@5": 0.022,
  "ndcg@10": 0.0273,
  "ndcg@20": 0.0327,
  "ndcg@50": 0.0425
}
```

## Code Framework Acknowledgement

Our implementation is based on the codebase of [STOSA](https://github.com/zfan20/STOSA?tab=readme-ov-file). We thank the authors for releasing their code.
