import json
import math
from math import log2, ceil
import random
import numpy as np
from Classifier import Classifier


class Node:
    obj_counter   = 0

    def __init__(self, parent = None, is_good_kid = False, arch_code_len = 0, is_root = False):
        # Note: every node is initialized as a leaf,
        # only non-leaf nodes are equipped with classifiers to make decisions
        if not is_root:
            assert type(parent) == type(self)

        self.parent        = parent
        self.is_good_kid   = is_good_kid
        self.ARCH_CODE_LEN = arch_code_len
        self.is_root       = is_root
        self.x_bar         = float("inf")
        self.n             = 0
        self.uct           = 0

        self.kids          = []
        self.bag           = {}

        # data for good and bad kids, respectively
        self.good_kid_data = {}
        self.bad_kid_data  = {}

        self.is_leaf       = True
        self.id            = Node.obj_counter
        self.layer        = ceil(log2(self.id + 2) - 1)
        self.classifier    = Classifier({}, self.ARCH_CODE_LEN, self.id)

        # insert current node into the kids of parent
        if parent is not None:
            self.parent.kids.append(self)
            if self.parent.is_leaf == True:
                self.parent.is_leaf = False
            assert len(self.parent.kids) <= 2

        Node.obj_counter += 1


    def clear_data(self):
        self.bag.clear()
        self.bad_kid_data.clear()
        self.good_kid_data.clear()


    def put_in_bag(self, net, maeinv):
        assert type(net) == type([])
        # assert type(maeinv) == type(float(0.1))
        net_k = json.dumps(net)
        self.bag[net_k] = (maeinv)


    def get_name(self):
        # state is a list of jsons
        return "node" + str(self.id)


    def pad_str_to_8chars(self, ins):
        if len(ins) <= 14:
            ins += ' ' * (14 - len(ins))
            return ins
        else:
            return ins


    def __str__(self):
        name = self.get_name()
        name = self.pad_str_to_8chars(name)
        name += (self.pad_str_to_8chars('lf:' + str(self.is_leaf)))

        name += (self.pad_str_to_8chars(' val:{0:.4f}   '.format(round(self.get_xbar(), 4))))
        name += (self.pad_str_to_8chars(' uct:{0:.4f}   '.format(round(self.get_uct(Cp=0.5), 4))))

        name += self.pad_str_to_8chars('n:' + str(self.n))
        name += self.pad_str_to_8chars('sp:' + str(len(self.bag)))
        name += (self.pad_str_to_8chars('g_k:' + str(len(self.good_kid_data))))
        name += (self.pad_str_to_8chars('b_k:' + str(len(self.bad_kid_data))))

        parent = '----'
        if self.parent is not None:
            parent = self.parent.get_name()
        parent = self.pad_str_to_8chars(parent)

        name += (' parent:' + parent)

        kids = ''
        kid = ''
        for k in self.kids:
            kid = self.pad_str_to_8chars(k.get_name())
            kids += kid
        name += (' kids:' + kids)

        return name


    def get_uct(self, Cp):
        if self.is_root and self.parent == None:
            return float('inf')
        if self.n == 0:
            return float('inf')
        coeff = math.pow(2, (6 - ceil(log2(self.id + 2)))) 
        if len(self.bag) < coeff * 20:
            return 0
        return self.x_bar + 2*Cp*math.sqrt(2*math.log(self.parent.n)/self.n)


    def get_xbar(self):
        return self.x_bar


    def train(self):
        if self.parent == None and self.is_root == True:
        # training starts from the bag
            assert len(self.bag) > 0
            self.classifier.update_samples(self.bag)
            self.good_kid_data, self.bad_kid_data = self.classifier.split_data()
        elif self.is_leaf:
            if self.is_good_kid:
                self.bag = self.parent.good_kid_data
            else:
                self.bag = self.parent.bad_kid_data
        else:
            if self.is_good_kid:
                self.bag = self.parent.good_kid_data
                self.classifier.update_samples(self.parent.good_kid_data)
                self.good_kid_data, self.bad_kid_data = self.classifier.split_data()
            else:
                self.bag = self.parent.bad_kid_data
                self.classifier.update_samples(self.parent.bad_kid_data)
                self.good_kid_data, self.bad_kid_data = self.classifier.split_data()
        if len(self.bag) == 0:
           self.x_bar = float('inf')
           self.n     = 0
        else:
           self.x_bar = np.mean(np.array(list(self.bag.values())))
           self.n     = len(self.bag.values())


    def predict(self):
        if self.parent == None and self.is_root == True and self.is_leaf == False:
            self.good_kid_data, self.bad_kid_data = self.classifier.split_predictions(self.bag)
        elif self.is_leaf:
            if self.is_good_kid:
                self.bag = self.parent.good_kid_data
            else:
                self.bag = self.parent.bad_kid_data
        else:
            if self.is_good_kid:
                self.bag = self.parent.good_kid_data
                self.good_kid_data, self.bad_kid_data = self.classifier.split_predictions(self.parent.good_kid_data)
            else:
                self.bag = self.parent.bad_kid_data
                self.good_kid_data, self.bad_kid_data = self.classifier.split_predictions(self.parent.bad_kid_data)


    def sample_arch(self):
        if len(self.bag) == 0:
            return None
        net_str = random.choice(list(self.bag.keys()))
        del self.bag[net_str]
        del self.parent.bag[net_str]
        return json.loads(net_str)
