from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from arch_generator import arch_generator
import json
import numpy as np
import copy  as cp
from collections import OrderedDict
from keras.optimizers import Adam
import numpy
from random import shuffle


class net_predictor:
    input_dim = 0
    output_dim = 100  # the resolution of accuracy is 100
    explore_depth = 0
    env_model = None
    ag = None

    def __init__(self):
        self.input_dim = 56
        self.env_model = self.build_env_model()

    def build_env_model(self):
        model = Sequential()
        model.add(
            Dense(512, input_dim=(self.input_dim), activation='relu', use_bias=True, kernel_initializer='RandomUniform',
                  bias_initializer='zeros'))
        model.add(
            Dense(2048, activation='relu', use_bias=True, kernel_initializer='RandomUniform', bias_initializer='zeros'))
        model.add(
            Dense(2048, activation='relu', use_bias=True, kernel_initializer='RandomUniform', bias_initializer='zeros'))
        model.add(
            Dense(512, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
        model.add(Dense(1, activation='sigmoid', use_bias=True))
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=0.0002), metrics=['mse'])

        # try:
        #     print('------loading from from file------')
        #     model.load_weights("env_model_weights.h5")
        # except:
        #     print('no saved model founded, start from scratch')
        return model

    def exp_to_train(self, trained_networks):
        exp_codes = []
        acc_codes = []
        if len(trained_networks) <= 0:
            return None, None
        for i in range(len(trained_networks)):
            network = trained_networks[i][0]
            acc = trained_networks[i][1]
            acc_codes.append(acc)
            # acc = int(np.around(acc * 100, 0))
            # print(acc)
            # if acc == 100:
            #     acc = 99
            # acc_code = np.zeros((1, 100))
            # acc_code[0, acc] = 1
            # acc_codes.append(acc_code)

            exp_codes.append(network)
        exp_codes = np.array(exp_codes)
        acc_codes = np.array(acc_codes)
        acc_codes = acc_codes.reshape(exp_codes.shape[0], -1)
        numpy.set_printoptions(threshold=numpy.nan)
        return exp_codes, acc_codes



    def predict(self, network):
        # Input: a list
        # Output: a float of accuracy
        network = predict_encoder(network=network)
        network = np.array(network)
        network = np.reshape(network, [1, len(network)])
        # print("network to predict:", network)
        accuracy = self.env_model.predict(network)
        # accuracy = float(np.argmax(accuracy)/100.0)
        # print("predicted accuracy", accuracy)
        return float(accuracy[0])

    def env_train(self, networks):
        concat_code = encoder(networks)
        # print("!!!!!env model training------------> start", len(exp_codes))
        # print(exp_codes)
        exp_codes, acc_codes = self.exp_to_train(concat_code)

        self.env_model.fit(exp_codes, acc_codes, batch_size=128, epochs=20, verbose=1)
        # save the model

        # print("!!!!!env model training------------> end", len(exp_codes))


def net_encoder(net):
    net_code =[]
    for i in range(len(net)-1):
        if net[i] == 'input':
            net_code.append(2)
        if net[i] == 'conv1x1-bn-relu':
            net_code.append(3)
        if net[i] == 'maxpool3x3':
            net_code.append(4)
        if net[i] == 'conv3x3-bn-relu':
            net_code.append(5)
        if net[i] == 'output':
            net_code.append(6)
    while len(net_code) < 7:
        net_code.append(9)
    return net_code

def predict_encoder(network):
    # Input: a OrderedDict
    # Output: a list

    net_arch = []
    node_list = network["node_list"]
    adj_mat = network["adj_mat"]
    net_code = net_encoder(node_list)

    for adj in adj_mat:
        for element in adj:
            net_arch.append(element)
    while len(net_arch) < 49:
        net_arch.append(0)
    for code in net_code:
        net_arch.append(code)

    return net_arch


def encoder(arch):
    concat_code = []
    for l, v in arch.items():
        net_info = []
        net_arch = []
        network = json.loads(l)
        node_list = network["node_list"]
        if node_list[-1] == 'term':
            node_list = node_list[:-1]
        adj_mat = network["adj_mat"]
        net_code = net_encoder(node_list)

        for adj in adj_mat:
            for element in adj:
                net_arch.append(element)
        while len(net_arch) < 49:
            net_arch.append(0)
        for code in net_code:
            net_arch.append(code)
        net_info.append(net_arch)
        net_info.append(v)
        concat_code.append(net_info)
    shuffle(concat_code)
    return concat_code
