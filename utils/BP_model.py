import math
import random



# 生成区间[a,b]内的随机数
def random_number(a, b):
    return (b - a) * random.random() + a


# 生成一个矩阵，大小为m*n,并且设置默认零矩阵
def makematrix(m, n, fill=0.0):
    a = []
    for i in range(m):
        a.append([fill] * n)
    return a


# 函数sigmoid(),这里采用tanh
def sigmoid(x):
    return math.tanh(x)


# 函数sigmoid的导数
def derived_sigmoid(x):
    return 1.0-x**2


class BPmodel(object):





    # 正向传播
    def update(self, inputs):
        if len(inputs) != self.num_in - 1:       #若输入值数量与输入层节点数不同
            raise ValueError('与输入层节点不同')

        #输入层
        for i in range(self.num_in - 1):
            self.active_in[i] = inputs[i]       #为输入层矩阵赋值

        #隐藏层
        for i in range(self.num_hidden - 1):
            sum = 0.0
            for j in range(self.num_in):
                sum = sum+self.active_in[j]*self.wight_in[j][i]     # 将输入层每个节电与对应的权重相乘
            self.active_hidden[i] = sigmoid(sum)           #使用sigmod函数为隐藏层赋值

        #输出层
        for i in range(self.num_out):
             sum=0
             for j in range(self.num_hidden):
                 sum = sum +self.active_hidden[j]*self.wight_out[j][i] #将隐藏层每个节点与对应的权重相乘获得输出值
             self.active_out[i] = sigmoid(sum)    # 同理

        return self.active_out[:]

    #误差的反向传播
    def eorrbackpropagate(self, targets, lr):  #lr为学习率
        if len(targets)!=self.num_out:
            raise ValueError('与输出层节点不同')

        #计算输出层的误差
        out_deltas = [0.0]*self.num_out
        for i in range(self.num_out):
            error = targets[i]-self.active_out[i]
            out_deltas[i] = derived_sigmoid(self.active_out[i])*error

        #计算隐藏层误差
        hidden_deltas =[0.0]*self.num_hidden
        for i in range(self.num_hidden):
            error = 0.0
            for j in range(self.num_out):
                error = error + out_deltas[j]*self.wight_out[i][j]
            hidden_deltas[i] = derived_sigmoid(self.active_hidden[i])*error

        #更新输出层的权重
        for i in range(self.num_hidden):
            for j in range(self.num_out):
                change = out_deltas[j]*self.active_hidden[i]
                self.wight_out[i][j]= self.wight_out[i][j]+lr*change

        #更新输入层权重
        for i in range(self.num_in):
            for j in range(self.num_hidden):
                change= hidden_deltas[j]*self.active_in[i]
                self.wight_in[i][j]=self.wight_in[i][j]+lr*change

        #总误差
        error = 0.0
        for i in range(len(targets)):
            error=error + 0.5*(targets[i]-self.active_out[i])**2
        return error





    def train(
            self,
            X,
            Y,
            num_in,
            num_hidden,
            num_out,
            itera,
            lr,
            ):
        self.num_in = num_in + 1  # 偏置节点
        self.num_hidden = num_hidden + 1  # 偏置节点
        self.num_out = num_out

        # 激活节点
        self.active_in = [1.0] * self.num_in
        self.active_hidden = [1.0] * self.num_hidden
        self.active_out = [1.0] * num_out

        # 创建权重矩阵
        self.wight_in = makematrix(self.num_in, self.num_hidden)
        self.wight_out = makematrix(self.num_hidden, self.num_out)


        if len(Y)!=len(X):
            raise ValueError('训练点数目没对上')

        for i in range(itera):
            error =0.0
            for j in range(len(X)):
                inputs = X[j]
                targets =Y[j]
                self.update(inputs)
                error =error +self.eorrbackpropagate(targets,lr)
            if i% 2==0:
                print('误差%-.5f' %error)




