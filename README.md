# EmbedSpot

EmbedSpot is an embedding-based retrievel model intended for personalized recommendations. 

## 1. Offline 

### 1.1 Conda environment

```bash
conda create -n embedspot python=3.7
conda activate embedspot
conda install --file requirements.txt --channel anaconda --channel conda-forge
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