# SILF  
This is the official PyTorch implementation of our TMM 2023 paper:  
> [Forgetting to Remember: A Scalable Incremental Learning Framework for Cross-Task Blind Image Quality   Assessment](https://arxiv.org/abs/2209.07126)  
> Rui Ma, Qingbo Wu, King Ngi Ngan, Hongliang Li, Fanman Meng, Linfeng Xu  
> *IEEE Transactions on Multimedia (TMM), 2023*  

## Getting Started

### File Organization

```
+-- ./Data
|   +-- [DatasetName]
|   |    +-- test
|   |    |   +-- test.txt
|   |    +-- train
|   |    |   +-- train.txt
|   |    +-- [DatasetName].txt
+-- main.py
+-- prune.py
+-- network.py
+-- dataset.py
```

### Requirements

- python 3.8.0
- torch 1.11.0
- torchvision 0.12.0
- scipy
- torchsummary
- tensorboardX

### Training and Testing

```
python network.py
python main.py
```

## Citation
If you find this project useful, please consider citing:

```
@article{ma2023forgetting,
  title={Forgetting to Remember: A Scalable Incremental Learning Framework for Cross-Task Blind Image Quality Assessment},
  author={Ma, Rui and Wu, Qingbo and Ngan, King Ngi and Li, Hongliang and Meng, Fanman and Xu, Linfeng},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  publisher={IEEE}
}
```
