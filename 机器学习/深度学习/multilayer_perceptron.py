from tkinter.messagebox import RETRY
from matplotlib.cbook import flatten
import numpy as np
from utils.features.prepare_for_training import prepare_for_training
from utils.hypothesis.hypothesis import sigmoid,sigmoid_gradient
# from sklearn.neural_network import _multilayer_perceptron # 多层感知机

class Multilayer_Perceptron:
    def __init__(self,data,labels,layers,normalize_data = False):
        data_processed = prepare_for_training(data, normalize_data=normalize_data)
        self.data  = data_processed
        self.labels = labels
        # 神经网络的层数 (1)输入层： 28*28*1 = 784  (2)隐藏层： 25个神经元  (3)输出层： 10 10分类任务
        self.layers = layers  # [784, 25, 10]
        self.normalize_data = normalize_data
        self.thetas = Multilayer_Perceptron.thetas_init(layers)
    
    def train(self,max_iters=1000,alpha = 0.1):
        """
        Func: 训练神经网络
        Args:
            max_iters (int): 最大迭代次数. Defaults to 1000.
            alpha (float): 学习率. Defaults to 0.1.
        """
        # 先对参数拉长
        unroll_thetas = Multilayer_Perceptron.thetas_unroll(self.thetas)
        Multilayer_Perceptron.gradient_descent(self.data,self.labels,self.layers,unroll_thetas,max_iters,alpha)

    # 对损失函数L进行梯度下降  w_t+1 = w_t - alpha*(L对w的偏导)_t (x)
    def gradient_descent(data,labels,layers,unroll_thetas,max_iters,alpha):
        optimized_theta = unroll_thetas # optimized_theta 优化后theta
        cost_history = [] # 记录损失值
        for _ in range(max_iters):
            # 计算损失
            cost = Multilayer_Perceptron.cost_function(data,labels,Multilayer_Perceptron.theta_roll(unroll_thetas,layers),layers)
            cost_history.append(cost)
            #计算梯度

            # 梯度更新 




    @staticmethod
    def cost_function(data,labels,thetas,layers):
        """
        计算损失值
        """
        num_layers = len(layers)
        num_samples = data.shape[0]
        num_labels = layers[-1]

        # 前向传播
        Multilayer_Perceptron.feedforward_propagation(data,thetas,layers)



    @staticmethod
    def feedforward_propagation(data,thetas,layers):
        """
        前向传播
        """
        num_samples = data.shape[0]
        num_layers = len(layers)


    @staticmethod
    def theta_roll(unrolled_thetas,layers):
        """
        把参数参数向量重新转化为矩阵，用来计算损失
        """
        num_layers = len(layers)
        thetas = {}
        unrolled_shift = 0  # 记录行theta处理到哪个位置了
        for layer_index in range(num_layers-1):
            in_count = layers[layer_index]
            out_count = layers[layer_index+1]
            # 矩阵的行和列
            theta_row = out_count
            theta_col = in_count + 1 
            theta_volume = theta_col * theta_row
            start_index = unrolled_shift
            end_index = unrolled_shift+theta_volume
            # 取出这一层未矩阵化的参数
            layer_unrolled_theta = unrolled_thetas[start_index:end_index]
            # reshape
            thetas[layer_index] = layer_unrolled_theta.reshape(theta_row,theta_col)
        return thetas

    @staticmethod
    def thetas_unroll(thetas):
        """
        把所有参数全部排列成一行,有利于参数更新
        """
        num_layers = len(thetas)
        unroll_theta = np.array([])
        for num_layer_index  in range (num_layers):
            # np.hstack 将多个数组沿水平方向（按列）堆叠，等价于 np.concatenate(axis=1)
            res_theta = np.hstack(res_theta,thetas[num_layer_index].flatten())
        return unroll_theta

    @staticmethod
    def thetas_init(layers):
        num_layers = len(layers)
        thetas = {}
        """
            thetas : 随机初始化所有参数
            layers : 网络层数
            for循环会执行两次, 25 * 785 , 10 * 26
        """
        for layer_index in range (num_layers -1):
            # 当前层的参数个数
            in_count = layers[layer_index]
            # 后层的参数个数
            out_count = layers[layer_index]
            # out_count,in_count 的顺序是因为是 (6,3)theta * x.T(3,1)
            # 加1 考虑的是偏置项
            thetas[layer_index] = 0.05 * np.random.rand(out_count,in_count+1)
        return thetas