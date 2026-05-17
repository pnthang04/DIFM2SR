<!-- Add banner here -->

# Beyond Feature Concatenation: Mutual Information-Driven Fusion for Multimodal Sequential Recommendation

## Requirements

- cython==0.29.20
- python==3.8.10
- pytorch==1.10.0
- pandas==2.0.3
- scipy==1.10.1
- colorama==0.4.6
- hyperopt==0.2.7

## Before Running

Please compile the cython codes before running:

```sh
python setup.py build_ext --inplace
```

## Training and Evaluation

To train and evaluate the model, run the following command:

```sh
python run_skrec.py
```