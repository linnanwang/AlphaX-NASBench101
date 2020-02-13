import numpy as np
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from collections import OrderedDict
from itertools import combinations
import copy as cp
import collections
import json
import copy
import random
from datetime import datetime



class arch_generator:
    #nasbench test
    operators     = [ 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3' ]
    MAX_NODES     = 6 #inclusive
    MAX_EDGES     = 9 #inclusive
    
    #maximal depth to go
    explore_depth = 5
    
    #common factors
    stride_low    = 1
    stride_up     = 1
    stride_step   = 1

    #convolution params
    filters_low   = 32
    filters_up    = 64 # 64 [filters_low, filters_up]
    filter_step   = 32

    kernel_low    = 2
    kernel_up     = 4
    kernel_step   = 2

    #pooling params
    pooling_oprs  = ['max', 'avg']

    pool_size_low = 1
    pool_size_up  = 2
    pool_step     = 1
    types = {'conv':0, 'pool':1, 'norm':2, 'act':3}
    #store ranges to generate actions
    params_range = { }
    
    layer_code_len = 4
    
    def __init__(self):
        self.params_range = {"filters":[self.filters_low, self.filters_up], "kernel_size":[self.kernel_low,  self.kernel_up] ,
            "pool_size":[self.pool_size_low, self.pool_size_up],
            "stride":[self.stride_low, self.stride_up] }


    def query_param_range(self, name):
        return self.params_range[name][0], self.params_range[name][1]
    
    def query_filter_step(self):
        return self.filter_step
    
    def query_kernel_step(self):
        return self.kernel_step
    
    def query_stride_step(self):
        return self.stride_step

    def query_pool_step(self):
        return self.pool_step
    
    def query_step(self, params):
        if params == 'filters':
            return self.query_filter_step()
        elif params == 'kernel_size':
            return self.query_kernel_step()
        elif params == 'stride':
            return self.query_stride_step()
        elif params == 'pool_size':
            return self.query_pool_step()
        return None
    
    def get_min_conv_layer(self, id):
        layer           = collections.OrderedDict()
        layer['id']     = id
        layer['type']   = 'conv'
        params          = collections.OrderedDict()
        params['filters']     = self.filters_low
        params['kernel_size'] = self.kernel_low
        params['stride']      = self.stride_low
        layer['params']       = params
        return collections.OrderedDict(sorted(layer.items()))
    
    def get_min_pool_layer(self, id):
        layer               = collections.OrderedDict()
        layer['id']         = id
        layer['type']       = 'pool'
        params              = collections.OrderedDict()
        params['pool_size'] = self.pool_size_low
        params['stride']    = self.stride_low
        layer['params']     = params
        return collections.OrderedDict(sorted(layer.items()))
    
    def get_norm_layer(self, id):
        layer             = collections.OrderedDict()
        layer['id']       = id
        layer['type']     = 'norm'
        return collections.OrderedDict(sorted(layer.items()))

    def get_activation(self, id):
        layer             = collections.OrderedDict()
        layer['id']       = id
        layer['type']     = 'act'
        return collections.OrderedDict(sorted(layer.items()))

    
    def norm(self, num, low, high):
        normalized = ((num - low)/(high - low) - 0.5) *2
        return normalized
    
    # coding rule: {type, filters, kernel/pool_size, stride}
    # 0 = N/A
    # types = {'conv':0, 'pool':1, 'norm':2, 'act':3}
    def code_conv_layer(self, layer):
        params      = layer['params']
        filteres    = params['filters']/10
        kernel_size = params['kernel_size']
        strides     = params['stride']
        return [0, filteres, kernel_size, strides]

    def code_pool_layer(self, layer):
        params      = layer['params']
        pool_size   = params['pool_size']
        strides     = params['stride']
        return [1, 0, pool_size, strides]
    
    def code_norm_layer(self, layer):
        return [2, 0, 0, 0]
    
    def code_act_layer(self, layer):
        return [3, 0, 0, 0]
    
    def decode_conv_layer(self, layer, input_shape = None):
        assert layer['type'] == 'conv'
        params      = layer['params']
        filters     = params['filters']
        kernel_size = params['kernel_size']
        strides     = params['stride']
        layer       = None
        if input_shape != None:
            layer = Conv2D(filters, (kernel_size, kernel_size), strides = (strides, strides), input_shape=input_shape, use_bias = True)
        else:
            layer = Conv2D(filters, (kernel_size, kernel_size), strides = (strides, strides), use_bias = True)
        return layer

    def decode_pool_layer(self, layer):
        assert layer['type'] == 'pool'
        params      = layer['params']
        #opr         = params['opr']
        opr         = 'max'
        ps          = params['pool_size']
        stride      = params['stride']
        layer       = None
        if opr == 'max':
            layer = MaxPooling2D(pool_size=(ps, ps), strides=(stride, stride), padding='valid')
        elif opr == 'avg':
            layer = AveragePooling2D(pool_size=(ps, ps), strides=(stride, stride), padding='valid')
        return layer

    def decode_norm_layer(self, layer):
        assert layer['type'] == 'norm'
        return BatchNormalization()

    def decode_act_layer(self, layer):
        assert layer['type'] == 'act'
        return Activation('relu')
    
    def get_first_layers(self):
        result = []

        params = collections.OrderedDict()
        for j in range(self.filters_low, self.filters_up+1, self.query_filter_step()):
            params['filters'] = j
            for k in range(self.kernel_low, self.kernel_up+1, self.query_kernel_step()):
                params['kernel_size'] = k
                network         = []
                layer           = collections.OrderedDict()
                layer['id']     = 0
                layer['params'] = params
                layer['type']   = 'conv'
                network.append( layer )
                result.append( cp.deepcopy(network) )
    
        #we use json dumps to maintain the order
        return result


    def code_network(self, network, explore_depth):
        network_code = []
        for i in range(0, len(network)):
            layer = network[i]
            if layer == 'term':
                layer_code = [-1, -1, -1, -1]
            else:
                assert i == int(layer['id'])
                if layer['type'] == 'conv':
                    layer_code = self.code_conv_layer(layer)
                elif layer['type'] == 'pool':
                    layer_code = self.code_pool_layer(layer)
                elif layer['type'] == 'norm':
                    layer_code = self.code_norm_layer(layer)
                elif layer['type'] == 'act':
                    layer_code = self.code_act_layer(layer)
            network_code.append(layer_code)
        for i in range(0, explore_depth - len(network_code) ):
            network_code.append( [0]*self.layer_code_len )
        network_code = np.concatenate( network_code[:] )
        return network_code

    def decode_network(self, net_code, explore_depth):
        net_code = net_code.reshape([-1, 4])
        print(net_code)
    
    def count_edges(self, adj_mat):
        ec = 0
        for l in adj_mat:
            ec += sum(l)
        return ec
            
    def get_potential_edges(self, adj_mat):
        candidates = []
        for i in range( 0, len(adj_mat) ):
            for j in range(0, len( adj_mat[i] ) ):
                if j > i: # only count the upper triangle
                    if adj_mat[i][j] == 0:
                        candidates.append( (i, j) )
        return candidates

    def get_actions(self, net ):
        # the actions will result in a vast number of
        # networks that do not exist in the truth table
        network = cp.deepcopy(net)
        actions =[]
        assert type(network) is type( collections.OrderedDict() )
        node_list = network["node_list"]
        adj_mat   = network["adj_mat"]
        # adding a new node
        actions.append("node:"+str(self.operators[0] ) )
        actions.append("node:"+str(self.operators[1] ) )
        actions.append("node:"+str(self.operators[2] ) )

        # mutate to the current graph by introducing an new edge
        edge_candidates = self.get_potential_edges(adj_mat)
        for e in edge_candidates:
            actions.append("edge:" + str(e) )
        # this will be a combination problem in python
        actions.append('term') # Terminal action
        return actions

    def print_adj_matrix(self, mat):
        for r in mat:
            print(r)

    def check_triangle_mat(self, mat):
        for ridx in range(0, len(mat)):
            for cidx in range(0, len(mat[ridx])):
                if ridx >= cidx and mat[ridx][cidx] > 0:
                    return False
        return True



    def get_next_network(self, net, action):
        # Input: net: a list
        # Output: a list
        network = cp.deepcopy(net)
        assert type(network) is type(collections.OrderedDict() )
        node_list = network["node_list"]
        adj_mat   = network["adj_mat"]
        node_c    = len(node_list)
        edge_c    = self.count_edges(adj_mat)
        if node_c > self.MAX_NODES:
            return None
        if edge_c > self.MAX_EDGES:
            return None
        if node_c == self.MAX_NODES and "node" in action:
            return None
        if edge_c == self.MAX_EDGES and "edge" in action:
            return None

        assert node_c <= self.MAX_NODES
        assert edge_c <= self.MAX_EDGES
        if 'edge' in action:
            node_list = network["node_list"]
            adj_mat   = network["adj_mat"]
            ridx = int(action[6])
            cidx = int(action[9])
            adj_mat[ridx][cidx] = 1
            network["adj_mat"]   = adj_mat
            network["node_list"] = node_list
            assert network["node_list"][0] == "input"
            assert network["node_list"][-1] == "output"
            assert type(network) is collections.OrderedDict
            return network
        elif 'node' in action:
            new_network = collections.OrderedDict()
            #wipe out all the existing edges
            node_type = action.split(":")[1]
            assert node_type in self.operators
            new_node_list = cp.deepcopy(node_list)
            new_node_list.insert(len(new_node_list) - 1, node_type)
            nodes_c = len(new_node_list)
            new_adj_mat = []
            for i in range(0, nodes_c):
                new_adj_mat.append( [0] * nodes_c )
            new_network["adj_mat"]  = new_adj_mat
            new_network["node_list"] = new_node_list
            assert new_network["node_list"][0] == "input"
            assert new_network["node_list"][-1] == "output"
            assert type(new_network) is collections.OrderedDict
            return new_network
        elif 'term' in action:
            assert network["node_list"][0] == "input"
            assert network["node_list"][-1] == "output"
            new_node_list = cp.deepcopy(network["node_list"])
            new_node_list.append("term")
            network["node_list"] = new_node_list
            assert type(network) is collections.OrderedDict
            return network
        else:
            raise "action must contains either object edge or node"
        return state

#ag = arch_generator()
#adj_mat   = [[0, 1, 1, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 1, 1],
#             [0, 0, 0, 0, 0, 1, 1],
#             [0, 0, 0, 0, 0, 0, 1],
#             [0, 0, 0, 0, 0, 0, 1],
#             [0, 0, 0, 0, 0, 0, 1],
#             [0, 0, 0, 0, 0, 0, 0]]
#node_list = ["input", ag.operators[0], ag.operators[0], ag.operators[0], ag.operators[0], ag.operators[0], "output"]
#network   = collections.OrderedDict({"adj_mat":adj_mat, "node_list":node_list })
#actions      = ag.get_actions( network )
#print("actions:", actions)
#for a in actions:
#    print("#"*10)
#    print("curt_network:")
#    ag.print_adj_matrix( network["adj_mat"] )
#    print("nodelist:", network["node_list"] )
#    next_network = ag.get_next_network( network, a )
#    print("-"*10)
#    print("next network, taking action", a)
#    print(next_network)
#    if a == "term":
#        print(ag.clean_term_network(next_network) )
##    ag.print_adj_matrix(next_network["adj_mat"] )
##    print("nodelist:", next_network["node_list"] )
#    print("\n\n")




