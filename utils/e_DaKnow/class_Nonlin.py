# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:45:18 2018

@author: yewenbin
"""

import numpy as np

class Nonlin(object):
    def __init__(self,x):
        self.x = x
    
    def nonlin(x ,deriv = False):
        if(deriv == True):
            return x*(1-x)
        return 1/(1+np.exp(-x))
    
if __name__ == '__main__':
    x = np.array([[0.5, 0.5, 0.3], [0.4, 0.6, 0.1]])
    non = Nonlin.nonlin(x)
    print(non)