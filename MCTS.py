import json
import collections
import copy as cp
import math
from arch_generator import arch_generator
from net_training import train_net
from net_predictor import net_predictor
from collections import OrderedDict
import os.path
import numpy as np
import time
import operator
import sys
import jsonpickle
import os
import random
from datetime import datetime


class Node:
    def __init__(self, state=None, x_bar=0, n=0, parent_str=None):
        assert state is not None
        assert parent_str is not None
        assert type(parent_str) is str
        print(type(state))
        print("curt state type:", type(state))
        assert type(state) is collections.OrderedDict

        self.x_bar = x_bar
        self.n = n
        self.parent_str = parent_str
        self.state = state

    def get_key(self):
        # state is a list of jsons
        return json.dumps(self.state, sort_keys=True)

    def get_state(self):
        return self.state

    def get_xbar(self):
        return self.x_bar

    def get_n(self):
        return self.n

    def get_parent_str(self):
        return self.parent_str

    def set_xbar(self, xb):
        self.x_bar = xb

    def set_n(self, n):
        self.n = n

    def set_parent(self, p):
        assert type(p) is str
        self.parent_str = p

    def print_str(self):
        print("-" * 10)
        print("state:", self.state)
        print("parent:", self.parent_str)
        print("xbar:", self.x_bar, " n:", self.n)

    def get_json(self):
        return json.dumps(self.state, sort_keys=True)

    def get_parent(self):
        return self.parent_str


class MCTS:
    #############################################
    nodes = collections.OrderedDict()  # actual MCTS tree
    dangling_nodes = collections.OrderedDict()  # this is to track the actually trained networks in random rollouts
    # curt state in String !!
    S = None
    Cp = 2
    simulations = 200
    arch_gen = None
    net_trainer = None
    net_predictor = None
    truth_table = None


    def __init__(self):
        self.arch_gen = arch_generator()
        self.net_predictor = net_predictor()
        self.net_trainer = train_net()
        self.loads_all_states()  # TODO: removed for develop
        self.trained_networks = {}
        self.simulated_networks = {}

        print("============search space start============")
        print("---------conv possibilities---------")
        print("filter range:",
              range(self.arch_gen.filters_low, self.arch_gen.filters_up + 1, self.arch_gen.filter_step))
        print("kernel range:", range(self.arch_gen.kernel_low, self.arch_gen.kernel_up + 1, self.arch_gen.kernel_step))
        print("stride range:", range(self.arch_gen.stride_low, self.arch_gen.stride_up + 1, self.arch_gen.stride_step))
        print("---------pool possibilities---------")
        print("kernel range:", range(self.arch_gen.kernel_low, self.arch_gen.kernel_up + 1, self.arch_gen.kernel_step))
        print("oprs types:", self.arch_gen.pooling_oprs, " currently default with MAX")
        print("stride range:", range(self.arch_gen.stride_low, self.arch_gen.stride_up + 1, self.arch_gen.stride_step))
        print("---------norm possibilities---------")
        print("default NORM in Keras")
        print("---------act  possibilities---------")
        print("RELU")
        print("=============search space end=============")
        print("")

        self.reset_to_root()

    def dump_all_states(self):
        node_path = 'nodes'
        with open(node_path, 'w') as outfile:
            json.dump(jsonpickle.encode(self.nodes), outfile)
        print("=>DUMP", len(self.nodes), " MCTS nodes")

        dangling_nodes_path = 'dangling_nodes'
        with open(dangling_nodes_path, 'w') as outfile:
            json.dump(jsonpickle.encode(self.dangling_nodes), outfile)
        print("=>DUMP", len(self.dangling_nodes), " dangling node")

    def loads_all_states(self):
        node_path = 'nodes'
        if os.path.isfile(node_path) == True:
            with open(node_path, 'r') as json_data:
                self.nodes = jsonpickle.decode(json.load(json_data, object_pairs_hook=OrderedDict))
        print("=>LOAD", len(self.nodes), " MCTS nodes")

        dangling_nodes_path = 'dangling_nodes'
        if os.path.isfile(dangling_nodes_path) == True:
            with open(dangling_nodes_path, 'r') as json_data:
                self.dangling_nodes = jsonpickle.decode(json.load(json_data, object_pairs_hook=OrderedDict))
        print("=>LOAD", len(self.dangling_nodes), " dangling nodes")

        # LOAD TRUTH TABLE
        truth_table_path = 'nasbench_dataset'
        with open(truth_table_path, 'r') as json_data:
            self.net_trainer.traing_mem = json.load(json_data, object_pairs_hook=OrderedDict)
        print("=>LOAD", len(self.net_trainer.traing_mem), " truth table entries")

    def create_dangling_node(self, state, rollout_from_str):
        # TODO this is to create a dangling node
        # to track a actually trained architecture in random rollouts
        # be sure to remove the last 'term'
        assert rollout_from is not None
        assert type(rollout_from_str) is str


        # the dangling node is create at func:evaluate_terminal

    def create_new_node(self, new_state=None, parent=None):
        # Expansion function
        # creating a regular node in a tree,
        # parent cannot be None, also the new node
        assert parent is not None
        assert new_state is not None

        new_state_str = json.dumps(new_state, sort_keys=True)

        parent_str = ""
        if parent != "ROOT":
            parent_str = json.dumps(parent, sort_keys=True)
            assert parent_str in self.nodes
        else:
            parent_str = "ROOT"

        # there are two possibilities:
        # the new node is a previous dangling node
        # the node is brand new
        xbar = 0
        # TODO once finish the actions
        xbar = self.evaluate(new_state, 2)
        assert xbar >= 0
        n = 1
        new_node = Node(new_state, xbar, n, parent_str)
        self.nodes[new_node.get_json()] = new_node

        if new_state_str in self.dangling_nodes:
            del self.dangling_nodes[new_state_str]

        return xbar

    def reset_to_root(self):
        # input->output is the ROOT
        network = collections.OrderedDict({"adj_mat": [[0, 0], [0, 0]], "node_list": ["input", "output"]})
        self.S = network
        state_str = self.get_state_str()
        if state_str not in list(self.nodes.keys()):
            self.create_new_node(self.S, "ROOT")  # state as a list is given
            # acc = self.net_trainer.train_net( self.get_state() )
            # acc = self.evaluate(state_str)
            # it is possible acc = 0
            # self.X_bar[state_str]      = acc
            # self.N[state_str]          = 1
            # TO DO: we should evaluate init state here.
        # assert state_str in self.experience.keys()
        assert state_str in list(self.nodes.keys())

    def print_nodes(self):
        print("\n$$$$$$$Nodes_Table>>>>>>>>>>>>start")
        id = 0
        for key, value in self.nodes.items():
            value.print_str()
            id += 1
        print("$$$$$$$Nodes_Table>>>>>>>>>>>>end\n")

    def print_dangling_nodes(self):
        print("\n#######Dangling_Nodes_Table>>>>>>>>>>>>start")
        id = 0
        for key, value in self.dangling_nodes.items():
            value.print_str()
            id += 1
        print("#######Dangling_Nodes_Table>>>>>>>>>>>>end\n")

    def get_state_str(self):
        return json.dumps(self.S, sort_keys=True)

    def get_state(self):
        return cp.deepcopy(self.S)

    def is_in_tree(self, state_str):
        # if state_str not in self.experience.keys():
        #     return False
        # if self.experience[state_str][0] is None:
        #     return False
        # return True
        assert type(state_str) is str
        if state_str not in self.nodes:
            return False
        return True

    def set_accuracy(self, state_str, accuracy):
        if state_str not in self.experience.keys():
            self.populate_experience(state_str, None, accuracy)
        else:
            self.experience[state_str][1] = accuracy

    def set_parent(self, state_str, parent):
        if state_str not in self.experience.keys():
            self.populate_experience(state_str, parent, None)
        else:
            self.experience[state_str][0] = parent

    def UCT(self, next_state):
        next_state_str = json.dumps(next_state, sort_keys=True)
        if next_state == None:
            # discourage the None nodes
            return 0
        else:
            if next_state_str not in self.nodes:
                # next state is a new node
                return float("inf")
            else:
                # next state is an existing node
                state_str = self.get_state_str()
                return self.nodes[next_state_str].get_xbar() + 2 * self.Cp * math.sqrt(
                    2 * math.log(self.nodes[state_str].get_n()) / self.nodes[next_state_str].get_n())

                # state_str      = self.get_state_str()
                # self.populate_N( next_state_str )
                # self.populate_N( state_str )
                # if self.get_N( next_state_str ) == 0:
                #     return float("inf")
                # else:
                #     assert state_str in self.N.keys()
                #     assert state_str in self.X_bar.keys()
                #     assert next_state_str in self.N.keys()

    def get_actions(self):
        # get legal actions
        return self.arch_gen.get_actions(self.get_state())

    def simulation(self, starting_net):
        # Input:  state (as a list)
        # Output: state (as a list). If the NN failed the model checking, return None.
        current_state = cp.deepcopy(starting_net)
        if current_state["node_list"][-1] == 'term':
            # If the current state is a terminal then return itself.
            return current_state
        counter = 0
        while True:
            rand_action = np.random.choice(self.arch_gen.get_actions(current_state))
            next_rand_net = self.arch_gen.get_next_network(current_state, rand_action)
            next_rand_net_str = json.dumps(next_rand_net, sort_keys=True)
            if next_rand_net == None:  # next_rand_net is None if it failed the model.
                current_state = starting_net
                continue

            if rand_action == 'term':
                trainable_str = json.dumps(self.clean_term_network(next_rand_net), sort_keys=True)
                
                if trainable_str in self.net_trainer.traing_mem:
                    print("=>simulation", next_rand_net)
                    return next_rand_net
                else:
                    # reset
                    current_state = starting_net
                    counter += 1
                    if counter > 1000:
                        return None
                    else:
                        continue

            current_state = next_rand_net

    def evaluate_terminal(self, terminal_node, rollout_from_str=None):
        # Input: state (as a list)
        # Output: accuracy
        # print("evaluate_terminal:", terminal_node)
        # TODO: in net_training, we will implement a trainning memory
        # to track every trained networks, and their accuracies.
        term_state_str = json.dumps(terminal_node, sort_keys=True)
        assert terminal_node is not None
        assert terminal_node["node_list"][-1] is 'term'

        if rollout_from_str is None:
            # this is a regular terminal node in MCTS,
            # no need to put it into dangling nodes
            nn = self.clean_term_network(terminal_node)
            nn_str = json.dumps(nn, sort_keys=True)
            if nn_str in self.net_trainer.traing_mem:
                acc, is_found = self.net_trainer.train_net(nn)
                self.trained_networks[json.dumps(nn, sort_keys=True)] = acc
                return acc
            else:
                return 0
        else:
            assert type(rollout_from_str) is str
            # this is a dangling node in MCTS rollout, at least, for that particular moment
            nn = self.clean_term_network(terminal_node)
            acc, is_found = self.net_trainer.train_net(nn)
            self.trained_networks[json.dumps(nn, sort_keys=True)] = acc
            if term_state_str not in self.dangling_nodes:
                self.dangling_nodes[term_state_str] = Node(terminal_node, acc, 0, rollout_from_str)
            return acc

    def clean_term_network(self, network):
        adj_mat = cp.deepcopy(network["adj_mat"])
        node_list = cp.deepcopy(network["node_list"])
        node_list.pop()
        new_network = collections.OrderedDict()
        new_network["adj_mat"] = adj_mat
        new_network["node_list"] = node_list
        return new_network

    def evaluate(self, starting_net, num_dyna_sim=0):
        # Input: node
        # Output: Xbar(node)
        # num_dyna_sim: number of simulations using the dyna value
        # print("evaluate: ", starting_net)
        # print("type of starting_net = ", type(starting_net))
        assert starting_net is not None

        ############################
        # condition1: evaluating the terminal node
        if starting_net["node_list"][-1] == 'term':
            print("==sim>>evaluating a terminal node")
            # If the starting_net is a terminal node, then we evaluate the terminal.
            return self.evaluate_terminal(starting_net, None)

        ############################
        # condition2: conducting random rollouts
        terminal_node = self.simulation(starting_net)

        ############################
        ### using true accuracy  ###
        ############################
        print("==sim>>estimating x_bar with simulation")
        print('terminal_NN is', terminal_node)
        if terminal_node == None:
            return 0.1
        else:
            acc = self.evaluate_terminal(terminal_node, json.dumps(starting_net, sort_keys=True))
        return acc
        # TODO: in the first step, let's set sim = 0
        # assert num_dyna_sim == 0
        # if num_dyna_sim == 0:
        #     return acc

        ############################
        ###   using simulations  ###
        ############################

#        sim_acc = 0.0
#        counter = 0
#        # Using model with dyna
#        # TODO: we take the average of dyna-sim results as its representative.
#        for i in range(num_dyna_sim):
#            terminal_node = self.simulation(starting_net)
#            assert terminal_node is not None
#            terminal_NN = self.clean_term_network(terminal_node)  # Remove the 'term' from the list.
#            if terminal_NN is None:
#                counter += 1
#                if json.dumps(terminal_NN, sort_keys=True) in self.trained_networks:
#                    p_acc = self.trained_networks[json.dumps(terminal_NN, sort_keys=True)]
#                    print('already trained')
#                else:
#                    p_acc = self.net_predictor.predict(terminal_NN)
#                    true_acc = self.net_trainer.traing_mem[json.dumps(terminal_NN, sort_keys=True)]
#            sim_acc += p_acc
#            print('now sim_acc is', p_acc)
#        sim_acc = sim_acc / num_dyna_sim
#
#        return (acc + sim_acc) / 2.0  # TODO: we take the average of true experience and dyna-sim results.

    #     def simulation(self, starting_net):
    #         #print "simulation starting net:", starting_net
    #         current_state = cp.deepcopy(starting_net)
    #         simulation_accuracies = []
    #         for i in range(0, self.simulations):
    #             rand_action   = np.random.choice( self.arch_gen.get_actions( current_state, self.experience ) )
    #             next_rand_net = self.arch_gen.get_next_network( current_state, rand_action )
    #             acc           = self.net_predictor.predict(next_rand_net)
    #             #print "current=>", json.dumps(current_state), " action:", rand_action
    #             #print "next====>", json.dumps(next_rand_net), " accuracy:", acc
    #             current_state = next_rand_net
    #             simulation_accuracies.append(acc)
    #             if next_rand_net == None:
    #                 current_state = cp.deepcopy(starting_net)
    #         return np.median(simulation_accuracies)

    def set_state(self, state):
        self.S = cp.deepcopy(state)

    def backpropagate(self, state, sim_result):
        cur_state = cp.deepcopy(state)
        curt_state_str = json.dumps(cur_state, sort_keys=True)
        i = 0
        while True:
            i = i + 1
            assert curt_state_str in self.nodes
            parent_str = self.nodes[curt_state_str].get_parent()  # 0: parent, 1: accuracy
            assert curt_state_str is not parent_str
            if parent_str == "ROOT":
                break
            # print "parent_str =", parent_str
            assert parent_str in self.nodes
            # print curt_state_str, "=>updating=>", parent_str, self.X_bar[parent_str]
            self.nodes[parent_str].set_n(self.nodes[parent_str].get_n() + 1)  # self.N[ parent_str ] += 1
            new_xbar = float(1 / self.nodes[parent_str].get_n()) * float(sim_result) + float(
                self.nodes[parent_str].get_n() - 1) / float(self.nodes[parent_str].get_n()) * self.nodes[
                                                                                           parent_str].get_xbar()
            self.nodes[parent_str].set_xbar(new_xbar)
            # print "after update=>", parent_str, self.X_bar[parent_str], "$$$$$$$$$$$", i
            curt_state_str = parent_str

    def search(self):
        episode = 0
        step = 0
        for i in range(0, 90000000):

            #            self.print_nodes()
            #            self.print_dangling_nodes()
            print("\n")
            print("#" * 30)
            print("episode:", episode, " step:", step, " nodes#", len(self.nodes), " dangling nodes:",
                  len(self.dangling_nodes), " counter:", self.net_trainer.counter, " total_sampled:",
                  len(self.net_trainer.training_trace))
            print("#" * 30)
            actions = self.get_actions()
            UCT = [0] * len(actions)
            for idx in range(0, len(actions)):
                act = actions[idx]
                next_net = self.arch_gen.get_next_network(self.get_state(), act)
                UCT[idx] = self.UCT(next_net)
            # the UCT = 0 if next_net is None, which is the case when the depth of network exceed the predefined explorable depth.
            best_action = actions[np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]]
            next_net = self.arch_gen.get_next_network(self.get_state(), best_action)
            next_net_str = json.dumps(next_net, sort_keys=True)
            curt_state_str = self.get_state_str()
            print("taken action:", best_action)
            print("next network:", json.dumps(next_net, sort_keys=True))

            # back-propagate on 3 conditions:
            # 1. exceed the max exploratory depth---back-propogate 0
            # 2. terminal---train the network
            # 3. new node---evaluate the node
            # the network exceed the explorable depth
            if next_net is None:
                print(">>>>RESET: exceeding the exploratory depth")
                # TODO:Question shall I back-propogate here????
                assert curt_state_str in self.nodes
                self.backpropagate(self.get_state(), 0.0)
                self.reset_to_root()
                episode += 1
                step = 0
            else:
                # create a new node
                if not self.is_in_tree(next_net_str):
                    print(">>>>RESET: creating a new non-terminal node")
                    new_node_xbar = self.create_new_node(next_net, self.get_state())
                    if new_node_xbar > 0:
                        self.backpropagate(next_net, new_node_xbar)
                    self.reset_to_root()
                    episode += 1
                    step = 0
                else:
                    # an existing node in tree
                    # double check the network length is within the range
                    if best_action == 'term':
                        # If the agent reaches the terminal state go back to the root.
                        print(">>>>RESET: reaching terminal in the tree")
                        # we shall only back-propogate on legitimate nodes
                        assert next_net_str in self.nodes
                        acc = self.nodes[next_net_str].get_xbar()
                        self.nodes[next_net_str].set_n(self.nodes[next_net_str].get_n() + 1)
                        self.backpropagate(next_net, acc)
                        self.reset_to_root()
                        episode += 1
                        step = 0
                    else:
                        print(">>>>Step forward: ")
                        if len(next_net) > self.arch_gen.explore_depth:
                            print("curt net length:", len(next_net), "and max depth is:", self.arch_gen.explore_depth)
                            assert len(next_net) <= self.arch_gen.explore_depth
                        # step into the next state
                        self.set_state(next_net)
                        step += 1
            if step > 1000:
                self.reset_to_root()
                # we also need reset to ROOT if the action is terminal
            ################# meta-NN update #################
            # We want to feed the net_predictor with experience of
            # NNs with accuracy.
            # nn_experience = collections.OrderedDict()
            # for k in self.experience.keys():
            #     if self.experience[k][1] is not None:
            #         k_no_term = k.replace(", \"term\"", "")  # TODO: hacky way
            #         nn_experience[k_no_term] = self.experience[k]
            # self.print_experience()
            # self.print_X_bar_table()
            # print "N TABLE"
            # self.print_N_table()
            # print "X TABLE"
            # self.print_X_bar_table()
            # print "nn_experience =", nn_experience
#            if i % 5 == 0:
#                self.net_predictor.env_train(self.trained_networks)
#                self.net_trainer.print_best_traces()


# self.dump_all_states()


agent = MCTS()
print(agent.get_state())
agent.search()
