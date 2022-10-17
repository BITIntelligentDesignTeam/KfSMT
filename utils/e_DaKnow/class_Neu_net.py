# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:52:34 2018

@author: yewenbin
"""

import numpy as np
from utils.e_DaKnow.class_Nonlin import Nonlin


class Neu_Net(object):
    def __init__(self, x, syn0, syn1):
        self.x = x
        self.syn0 = syn0
        self.syn1 = syn1

    def Neu_net(self):
        l1 = Nonlin.nonlin(np.dot(self.x, self.syn0))
        l2 = Nonlin.nonlin(np.dot(l1, self.syn1))
        return l2


if __name__ == '__main__':

    Neu = Neu_Net.Neu_net(x, syn0, syn1)
    print(Neu)
