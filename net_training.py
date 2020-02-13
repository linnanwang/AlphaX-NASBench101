import numpy as np
import time

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from arch_generator import arch_generator
import sys
import copy
from datetime import datetime
import collections
import json
import operator
import os

from keras.layers.normalization import BatchNormalization
from keras.layers import Activation

class train_net:
    best_trace      = collections.OrderedDict()
    acc_trace       = collections.OrderedDict()
    traing_mem      = collections.OrderedDict()
    training_trace  = collections.OrderedDict()
    target_str      = None
    best_accuracy   = 0
    x_train         = []
    y_train         = []
    x_test          = []
    y_test          = []
    sgd             = None
    counter         = 0
    num_classes     = 0
    ag = arch_generator()
    
    def print_best_traces(self):
        print("%"*20)
        print("=====> best accuracy so far:", self.best_accuracy)
        sorted_best_traces = sorted(self.best_trace.items(), key=operator.itemgetter(1))
        for item in sorted_best_traces:
            print(item[0],"==>", item[1])
        for item in sorted_best_traces:
            print(item[1])
        print("%"*20)
    
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test  = x_test.astype('float32')
        self.x_train = x_train / 255.0 #normalize
        self.x_test  = x_test / 255.0  #normalize
        self.y_train = np_utils.to_categorical( y_train )
        self.y_test  = np_utils.to_categorical( y_test )
        self.num_classes = self.y_test.shape[1]
        # 7 nodes
        t_adj_mat  = [[0, 1, 1, 0, 0, 1, 1],
                     [0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0]]
        t_node_list = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'output']
        # 6 nodes
        # t_adj_mat  = [[0, 1, 1, 1, 1, 1],
        #              [0, 0, 0, 0, 1, 0],
        #              [0, 0, 0, 1, 0, 0],
        #              [0, 0, 0, 0, 1, 0],
        #              [0, 0, 0, 0, 0, 1],
        #              [0, 0, 0, 0, 0, 0]]
        # t_node_list =  ['input', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'output']
        #5 nodes
#        t_adj_mat  = [[0, 1, 1, 1, 1],
#                      [0, 0, 1, 1, 0],
#                      [0, 0, 0, 1, 0],
#                      [0, 0, 0, 0, 1],
#                      [0, 0, 0, 0, 0]]
#        t_node_list = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'output']
        #4 nodes
#        t_adj_mat = [[0, 1, 0, 1],
#                     [0, 0, 1, 0],
#                     [0, 0, 0, 1],
#                     [0, 0, 0, 0]]
#        t_node_list = ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'output']
        #3 nodes
#        t_adj_mat   = [[0, 1, 1], [0, 0, 1], [0, 0, 0]]
#        t_node_list = ['input', 'conv3x3-bn-relu', 'output']

        target      = collections.OrderedDict( {"adj_mat": t_adj_mat, "node_list":t_node_list } )
        self.target_str = json.dumps( target, sort_keys = True )

    def check_between_conv_layers( self, layers ):
        #between two conv, no more than 1 pool, 1 act and 1 norm
        counter = {}
        counter['conv'] = 0
        counter['pool'] = 0
        counter['norm'] = 0
        counter['act']  = 0
        curt_conv_layer_id = 0 # assume the first layer is conv
        for i in range( 1, len(layers) ):
            next_conv_layer = i
            while i < len(layers) and layers[i]['type'] != 'conv':
                counter[ layers[i]['type'] ] += 1
                i = i+1
            if counter['norm'] > 1 or counter['act'] > 1 or counter['pool'] > 1:
                return False
            counter['conv'] = 0
            counter['pool'] = 0
            counter['norm'] = 0
            counter['act']  = 0
        return True

    def duplication_check( self, layers ):
        # only allows two conv layer to be adjacent
        # conv, pool, norm, act
        for i in range(1, len(layers)):
            if layers[i-1]['type'] == layers[i]['type'] and layers[i]['type'] != 'conv':
                return False
        return True
    
    def design_check(self, layers):
        #True:  pass the design check
        #False: fail the design check
        #guarantee conv to be the first
        if layers[0]['type'] != 'conv':
            return False
        if self.duplication_check( layers ) == False:
            return False
        if self.check_between_conv_layers( layers ) == False:
            return False
        return True
    
    def model_check(self, model):
        layers = model.layers
        for l in layers:
            params_count = l.count_params()
            if params_count > 400000:
                return False
        return True
    
    def state_to_model(self, state):
        model  = Sequential()
        layers = [None] * len(state)
        for layer in state: #ensure the order
            layer_id         = layer['id']
            layers[layer_id] =  layer
        print("##prepare to train=>")
        for l in layers:
            print(l)

        if self.design_check(layers) == False:
            print("design failed checking")
            return None

        for i in range(0, len(state)):
            try:
                layer        = layers[i]
                layer_id     = layer['id']
                layer_type   = layer['type']
                layer_model  = None
                if layer_type == 'conv':
                    if layer_id == 0:
                        layer_model = self.ag.decode_conv_layer( layer, (3, 32, 32) )
                        #this is force to have conv->act->bn blocks
                    else:
                        layer_model = self.ag.decode_conv_layer( layer )
                elif layer_type == 'pool':
                    layer_model = self.ag.decode_pool_layer( layer )
                elif layer_type == 'act':
                    layer_model = self.ag.decode_act_layer( layer )
                elif layer_type == 'norm':
                    layer_model = self.ag.decode_norm_layer( layer )
                model.add( layer_model )
                #TODO only convolution layer
                model.add( Activation('relu') )
                model.add( BatchNormalization() )
            except Exception as e:
                print(e)
                os._exit(0)
                return None
        model.add(Flatten())
        model.add( Dense(self.num_classes, activation='softmax', use_bias = True) )
        print(model.summary())
        
        # if self.model_check(model) == False:
        #     print("model failed checking")
        #     return None
        return model
   
    def train_net(self, network):
        #this state has been cleaned from term
        network_str = json.dumps( network, sort_keys = True )
        assert network_str in self.traing_mem
        is_found = False
        acc = self.traing_mem[network_str]
        if network_str not in self.training_trace:
            self.training_trace[network_str] = acc
        self.counter += 1
        if acc > self.best_accuracy:
            print("@@@update best state:", network)
            print("@@@update best acc:", acc)
            self.best_accuracy = acc
            item = [acc, self.counter]
            self.best_trace[network_str] = item
            print("target str:", self.target_str)
        if self.counter % 1000 == 0:
            sorted_best_traces = sorted(self.best_trace.items(), key=operator.itemgetter(1))
            final_results = []
            for item in sorted_best_traces:
                final_results.append( item[1] )
            final_results_str = json.dumps(final_results)
            filename = "results/result_"+str(self.counter)
            with open(filename, "a") as f:
                f.write(final_results_str + '\n')

        return acc, is_found
