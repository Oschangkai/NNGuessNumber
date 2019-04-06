# -*- coding: utf-8 -*-
import numpy as np

# For Mac
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

import copy

import subprocess, os, platform

# 在機器學習中，大家習慣使用矩陣存
def ConvertToBitArray(value, num_bit):

    val = copy.deepcopy(value)

    bit_array = np.zeros((num_bit, 1))
    i = 0

    while val > 0:
        if(val & 0x01):
            bit_array[i] = 1
        val >>= 1
        i += 1

    return bit_array


def Popcount(value):

     val = copy.deepcopy(value)

     count = 0
     i = 0

     while val > 0:
         if(val & 0x01):
             count += 1
         val >>= 1
         i += 1

     return count


def BitToMaxNumber(bits):

    maxNumber = 0
    while(bits > 0):
        maxNumber |= 1
        maxNumber <<= 1
        bits -= 1

    maxNumber >>= 1

    return maxNumber


class NeuralNetwork(object):

    def __init__(self, bitNumber, num_neurons):
        self.bitNumber = bitNumber
        self.num_neurons = num_neurons
        self.outputNumber = self.bitNumber + 1

        self.W1 = np.random.uniform(-1, 1, (num_neurons, bitNumber))
        self.b1 = np.random.uniform(-1, 1, (num_neurons, 1))

        self.W2 = np.random.uniform(-1, 1, (self.outputNumber, num_neurons))
        self.b2 = np.random.uniform(-1, 1, (self.outputNumber, 1))
        # print('W1 = ',self.W1, 'b1 = ',self.b1, 'W2 = ',self.W2, 'b2 = ',self.b2)

    def Softmax(self, val):
        exps = np.exp(val - np.max(val))
        return exps / np.sum(exps, axis=0)

    # Activation Function: ReLU - 去除小於 0 的結果
    def ReLU(self, val):
        return (val > 0) * val

    def deRelu(self, z):
        return (z > 0) * 1

    def cross_entropy(self, y) :
        #return -np.sum( y * np.log(self.out))
        for i in range(y.size):
            if (0 != y[i]):
                return -np.log(self.out[i])
        pass

    def Forward(self, x):

        self.x = x

        self.z1 = np.dot(self.W1, self.x)
        self.z1 += self.b1
        self.z2 = self.ReLU(self.z1)

        self.z3 = np.dot(self.W2, self.z2)
        self.z3 += self.b2

        self.out = self.Softmax(self.z3)

        return self.out

    def Backward(self, y):

        loss = self.cross_entropy(y)
        out_error = self.out - y

        z3_error = out_error
        z2_error = self.W2.T.dot(z3_error)
        z1_error = z2_error * self.deRelu(self.z2)

        z3_W_delta = z3_error.dot(self.z2.T)
        z3_b_delta = z3_error

        z1_W_delta = z1_error.dot(self.x.T)
        z1_b_delta = z1_error

        lr = 5e-3
        self.W2 -= z3_W_delta * lr
        self.W1 -= z1_W_delta * lr

        self.b2 -= z3_b_delta * lr
        self.b1 -= z1_b_delta * lr

        return loss

if __name__ == "__main__":

    num_bits = 8 # 二進位的位數
    num_neurons = 64 # 神經元數量
    max_epoch = 1000 # 最多 Fitting 幾次

    num_training_samples = 100 * num_bits # 樣本數
    max_value = BitToMaxNumber(num_bits) # Bit to Dec


    print("bit number = %d, max value = %d"%(num_bits, max_value))
    print("num_training_samples = %d"%num_training_samples)

    x = np.random.randint(0, high = max_value,
                            size = (num_training_samples, 1))
#   Train
    nn = NeuralNetwork(num_bits, num_neurons)

    loss = []
    error_rate = []
    for epoch in range(max_epoch):
        err_count = 0
        for i in range(np.size(x, 0)):

            y = Popcount(x[i])
            x_array = ConvertToBitArray(x[i], num_bits)

            #print("value = %d, popcount = %d "%( x_array[i], y))
            out_array = nn.Forward(x_array)

            jj = 0
            for j in range(out_array.size):
                if(out_array[j] > out_array[jj]):
                    jj = j
             # print("predict bits = %d "% jj);

            #np.set_printoptions(suppress = True)
            #print("\npredicte = \n%s" % np.reshape(out_array, (bitNumber + 1, 1)) )

            yy = np.zeros((num_bits + 1, 1))
            yy[int(y)] = 1.0

            if(int(y) != jj):
                err_count += 1

            loss.append( nn.Backward(yy))

        error_rate.append((100.0 * err_count) / np.size(x, 0))

        # %4.3f => 總共有四位數，小數後共有三位
        print("epoch = %d, training error rate = %4.3f%%, loss = %6.5f"
                % (epoch, error_rate[-1], loss[-1]) )

#       兩個停止 epoch 的條件
#       1. 
        if(abs(loss[-1] - loss[-2]) <= 1e-4 and loss[-1] <= 1e-4):
            break
#       2. 誤差值 < 0.2
        if(error_rate[-1] <= 0.2):
            break;
# END FOR

# 畫圖
    step = int(num_training_samples)
    loss = loss[0:-1:step]

    fig = plt.figure(figsize=(8, 6))
    title_str = str.format("number of bits = %d, neurons = %d \n"
                             "training samples = %d"
                             %(num_bits, num_neurons,
                             num_training_samples))
    fig.suptitle(title_str, fontsize = 14)

    x1 = np.linspace(0, len(loss), len(loss) )

    ax1 = plt.subplot(2, 1, 1)

    ax1.semilogy( x1, loss, lw = 2)
    ax1.legend(["log(loss)"])
    ax1.grid(True)

    ax2 = plt.subplot(2, 1, 2)
    ax2.set_xlabel("epoch")
    x2 = np.linspace(0, len(error_rate), len(error_rate) )
    ax2.plot( x2,
             error_rate , lw = 2, color='magenta')
    ax2.legend(["training data error rate(%)"])
    ax2.grid(True)


#   猜數字
    num_test_sample = 100

    err_count = 0

    for i in range(num_test_sample):
        x = np.random.randint(0, high = max_value)
        y = Popcount(x)
        x_array = ConvertToBitArray(x, num_bits)

        out_array = nn.Forward(x_array)

        jj = 0
        for j in range(out_array.size):
            if(out_array[j] > out_array[jj]):
                jj = j # 找出機率最大的那個，1個1的機率，還是2個1的機率？
        if(int(y) != jj):
            err_count += 1 # 比較1有多少個，如果不一樣多，代表數字錯誤

    print("prediction error rate in test = %f %%"
            % (100*err_count/num_test_sample))

#   顯示結果圖
    plt.savefig('fig.png')
    if platform.system() == 'Darwin':       # macOS
        subprocess.call(('open', 'fig.png'))
    elif platform.system() == 'Windows':    # Windows
        # os.startfile('fig.png')
        plt.show()
    else:                                   # linux variants
        subprocess.call(('xdg-open', 'fig.png'))