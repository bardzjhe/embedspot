# EmbedSpot

EmbedSpot is an embedding-based retrievel model intended for personalized recommendations. 

## Prerequisites


1. Anaconda

2. Docker
3. GCC Compiler

## 1. Offline
### 1.1 Conda environment

```bash
conda create -n embedspot python=3.7
conda activate embedspot

# For GPU
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia 
# For CPU
#conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 cpuonly -c pytorch
conda install --file requirements.txt --channel anaconda --channel conda-forge

# preprocess dataset to merge three .dat files
cd examples/matching/data/ml-1m
python preprocess_ml.py
```

### 1.2 Offline batch inference
In EmbedSpot, we conduct preprocessing. For positive sampling, we take advantage of both implicit feedback (click) and explicit feedback (ratings)
Among the items that users have interacted with, ratings greater than 3 are labeled as positive samples,
when negative sampling follows the popularity-based sampling strategies as shown in our report. 
```python
python setup.py install
```

## 1.Offline to Online



## Acknowledge

1. Torch-RecHub ([GitHub](https://github.com/datawhalechina/torch-rechub))
2. recsys_pipeline ([GitHub](https://github.com/akiragy/recsys_pipeline/tree/master))