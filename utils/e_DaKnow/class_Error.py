import numpy as np
from utils.e_DaKnow.class_Neu_net import Neu_Net

class Error(object):
    def __init__(self,Hid_layer, syn0, syn1, x, y):
        self.Hid_layer = Hid_layer
        self.syn0 = syn0
        self.syn1 = syn1
        self.x = x
        self.y = y

    
    def error(self):

        l2_0 = Neu_Net(self.x,self.syn0,self.syn1)    
        l2 = l2_0.Neu_net()
        l2 = np.array(l2)
        l2_error = abs(self.y - l2) 
        error =np.mean(l2_error) 
        
        return error


if __name__=='__main__': 
    
    def nonlin(x ,deriv = False):
        if(deriv == True):
            return x*(1-x)
        return 1/(1+np.exp(-x))
    
    def Neu_net(x,syn0,syn1):
        l1 = nonlin(np.dot(x,syn0))
        l2 = nonlin(np.dot(l1,syn1))
        return l2
    
    error = Error.error(Hid_layer,syn0,syn1,x,y)
    print(error)