# AlphaX-NASBench101
AlphaX is a new Neural Architecture Search (NAS) agent that uses MCTS for efficient model architecture search with Meta-DNN as a predictive model to estimate the accuracy of a sampled architecture. Compared with Random Search, AlphaX builds an online model which guides the future search, compared to greedy methods, e.g. Q-learning, Regularized Evolution or Top-K methods, AlphaX dynamically trades off exploration and exploitation and can escape from locally optimal solutions with fewer number of search trials. For details of AlphaX, please refer to [AlphaX: eXploring Neural Architectures with Deep Neural Networks and Monte Carlo Tree Search](https://arxiv.org/abs/1805.07440).

This repository hosts the implementation of AlphaX for searching on a design domain defined by NASBench-101. NASBench-101 is a NAS dataset that contains 420k+ networks with their actual training, validation accuracies. For details of NASBench-101, please check [here](https://github.com/google-research/nasbench).

AlphaX is 2.8x and 3.0x faster than Regularized Evolution and Random Search on NASBench dataset (constrain the nodes <= 6 for 200 trails).

<img src='https://github.com/linnanwang/AlphaX-CVPR2019/blob/master/fig/mcts_speed_nasbench.png?raw=true' width="400">

This is how MCTS progressively probes the search domain. Each node represents an MCTS state; the node color reflects its value, i.e. accuracy, indicating how promising a search branch.

<img src='https://github.com/linnanwang/AlphaX-CVPR2019/blob/master/fig/mcts_viz.png?raw=true' width="400">

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

**MCTS without meta_DNN assisted** 
```
python MCTS.py
```
* Total numbers of sampled networks, current best accuracy and state will list on the screen. 

**MCTS with meta_DNN assisted** 
```
python MCTS_metaDNN.py
```
* Total numbers of sampled networks, current best accuracy/state and meta_DNN simulated accuracy will list on the screen.

**Note**:  Since meta_DNN is going to learn the prediction of network accuracy which needs sometime for training, MCTS with meta_DNN will take extra time in searching process. The default maximum nodes number is 6 in our program.  If you would like to change the maximum nodes number,  you can follow the steps as below. 
- Please see arch_generator.py. Change the MAX_NODES from 6 to the number you want(line 20). 
```python
class arch_generator:
#nasbench test
operators     = [ 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3' ]
MAX_NODES     = 6 #inclusive
MAX_EDGES     = 9 #inclusive
```
- Also see net_training.py.  Comment the line from 74-80 and uncomment the codes represent the node number you want(line 64-96).
```python
# 7 nodes
#        t_adj_mat  = [[0, 1, 1, 0, 0, 1, 1],
#                      [0, 0, 0, 0, 0, 1, 0],
#                      [0, 0, 0, 1, 0, 0, 0],
#                      [0, 0, 0, 0, 1, 0, 0],
#                      [0, 0, 0, 0, 0, 1, 0],
#                      [0, 0, 0, 0, 0, 0, 1],
#                      [0, 0, 0, 0, 0, 0, 0]]
#        t_node_list = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'output']
# 6 nodes
t_adj_mat  = [[0, 1, 1, 1, 1, 1],
[0, 0, 0, 0, 1, 0],
[0, 0, 0, 1, 0, 0],
[0, 0, 0, 0, 1, 0],
[0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0]]
t_node_list =  ['input', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'output']
#5 nodes
#        t_adj_mat  = [[0, 1, 1, 1, 1],
#                      [0, 0, 1, 1, 0],
#                      [0, 0, 0, 1, 0],
#                      [0, 0, 0, 0, 1],
#                      [0, 0, 0, 0, 0]]
#        t_node_list = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'output']
```
## Results



## Citation
If you use these codes in your research, please cite:

```
@ARTICLE{DBLP:journals/corr/abs-1805-07440,
author = {Linnan Wang and Yiyang Zhao and Yuu Jinnai and Rodrigo Fonseca},
title = "{AlphaX: eXploring Neural Architectures with Deep Neural Networks and Monte Carlo Tree Search}",
journal = {arXiv e-prints},
year = "2019",
eid = {arXiv:1805.07440}
}
```







