# AlphaX-NASBench101
AlphaX is a new Neural Architecture Search (NAS) agent that uses MCTS for efficient model architecture search with Meta-DNN as a predictive model to estimate the accuracy of a sampled architecture. Compared with Random Search, AlphaX builds an online model which guides the future search, compared to greedy methods, e.g. Q-learning, Regularized Evolution or Top-K methods, AlphaX dynamically trades off exploration and exploitation and can escape from locally optimal solutions with fewer number of search trials. For details of AlphaX, please refer to [AlphaX: eXploring Neural Architectures with Deep Neural Networks and Monte Carlo Tree Search](https://arxiv.org/abs/1805.07440).

This repository hosts the implementation of AlphaX for searching on a design domain defined by NASBench-101. NASBench-101 is a NAS dataset that contains 420k+ networks with their actual training, validation accuracies. For details of NASBench-101, please check [here](https://github.com/google-research/nasbench).

In 200 search trails, AlphaX is 2.8x and 3.0x faster than Regularized Evolution and Random Search on NASBench dataset.

<img src='https://github.com/linnanwang/AlphaX-CVPR2019/blob/master/fig/mcts_speed_nasbench.png?raw=true' width="400">

This is how MCTS progressively probes the search domain. Each node represents an MCTS state; the node color reflects its value, i.e. accuracy, indicating how promising a search branch.

<img src='https://github.com/linnanwang/AlphaX-CVPR2019/blob/master/fig/mcts_viz.png?raw=true' width="400">

Please cite our work, if it helps your research ;)
```
@article{wang2018alphax,
  title={AlphaX: eXploring Neural Architectures with Deep Neural Networks and Monte Carlo Tree Search},
  author={Wang, Linnan and Zhao, Yiyang and Jinnai, Yuu},
  journal={arXiv preprint arXiv:1805.07440},
  year={2018}
}
```

## Requirements
```
Python >= 3.5.5, numpy >= 1.9.1, keras >= 2.1.6, jsonpickle
```

## Setup

1.  Clone this repo.

```
git clone git@github.com:linnanwang/AlphaX-NASBench101.git
cd AlphaX-NASBench101
```

2. (optional) Create a virtualenv for this library.

```
virtualenv --system-site-packages -p python3 ./venv
source venv/bin/activate
```

3. Install the project along with dependencies.

```
pip install numpy
pip install keras
pip install jsonpickle
```

## Download the dataset

The full NASBench dataset in our format is at [here](https://drive.google.com/file/d/100xB4Mj7Hc5I0ljVPo7ATmC2kfhytHuN/view?usp=sharing). Please place the dataset into the same directory of AlphaX-NASBench101.

## Usage

After prepared all dependencies, execute the following commands to start the search:

**MCTS without meta_DNN** (Fast on CPU) 
```
python MCTS.py
```
**MCTS with meta_DNN assisted** (Slow, and please run on GPU.)
```
python MCTS_metaDNN.py
```

**Note**: meta_DNN requires training to predict the accuracy of a unseen architecture. MCTS with meta_DNN will take extra time in searching process.

## Changing the size of search domain
By default, we constrain the nodes <= 6, that constitus of 60000+ valid networks. The following steps illustate how to expand or constrain the search domain.

- In arch_generator.py, changing the MAX_NODES to any in [3, 4, 5, 6, 7] (line 20). NASBench-101 provides all the networks up to 7 nodes.
```python
class arch_generator:
MAX_NODES     = 6 #inclusive
MAX_EDGES     = 9 #inclusive
```
- Also lines from 74-80 in net_training.py defines the target. The search stops once it hits the target. The target consists of two parts, the adjacent matrix and the node list. Please change it to a different target after you changing the maximal nodes.
```python
# 6 nodes
t_adj_mat  = [[0, 1, 1, 1, 1, 1],
[0, 0, 0, 0, 1, 0],
[0, 0, 0, 1, 0, 0],
[0, 0, 0, 0, 1, 0],
[0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0]]
t_node_list =  ['input', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'output']
```







