import numpy as np
from utils.features.prepare_for_training import prepare_for_training
from utils.hypothesis.hypothesis import sigmoid,sigmoid_gradient
# from sklearn.neural_network import _multilayer_perceptron # 多层感知机

class Multilayer_Perceptron:
    def __init__(self,data,labels,layers,normalize_data = False):
        """
        # 神经网络的层数
        (1)输入层： 28*28*1 = 784  (2)隐藏层： 25个神经元  (3)输出层： 10 10分类任务
        第一层是输入层 ,输入像素点个数  28*28*1   长 宽 颜色通道--不可改
        第二层:隐层神经元个数 25个(把784特征转化为25维向量)  --可改
        第三层:分类层,10分类任务
        """ 
        data_processed = prepare_for_training(data, normalize_data=normalize_data)[0]
        self.data  = data_processed
        self.labels = labels
        self.layers = layers  # [784, 25, 10]
        self.normalize_data = normalize_data
        self.thetas = Multilayer_Perceptron.thetas_init(layers)
    
    def train(self,max_iters=1000,alpha = 0.1):
        """
        Func: 训练神经网络
        Args:
            max_iters (int): 最大迭代次数. Defaults to 1000.
            alpha (float): 学习率. Defaults to 0.1.
            theta_unroll: 参数向量型式
            theta_roll:  参数矩阵型式
        """
        # 先对参数拉长
        unroll_thetas = Multilayer_Perceptron.thetas_unroll(self.thetas)
        # 执行梯度下降
        (optimized_theta_unroll,cost_history) = Multilayer_Perceptron.gradient_descent(self.data,self.labels,self.layers,unroll_thetas,max_iters,alpha)
        optimized_theta_roll = Multilayer_Perceptron.theta_roll(optimized_theta_unroll,self.layers)
        self.thetas = optimized_theta_roll
        return self.thetas,cost_history

    def predict(self, data):
        data_processed = prepare_for_training(data,normalize_data = self.normalize_data)[0]
        num_examples = data_processed.shape[0]
        predictions = Multilayer_Perceptron.feedforward_propagation(data_processed,self.thetas,self.layers)
        return np.argmax(predictions,axis=1).reshape((num_examples,1))#返回最大概率值

    # 对损失函数L进行梯度下降  w_t+1 = w_t - alpha*(L对w的偏导)_t (x)
    def gradient_descent(data,labels,layers,unroll_thetas,max_iters,alpha):
        optimized_theta = unroll_thetas # optimized_theta 优化后theta
        cost_history = [] # 记录损失值
        for _ in range(max_iters):
            # 计算损失
            cost = Multilayer_Perceptron.cost_function(data,labels,Multilayer_Perceptron.theta_roll(optimized_theta,layers),layers)
            cost_history.append(cost)
            #计算梯度
            theta_gradient = Multilayer_Perceptron.gradient_step(data,labels,optimized_theta,layers)
            # 梯度更新 
            optimized_theta = optimized_theta - theta_gradient*alpha
        return optimized_theta,cost_history


    @staticmethod
    def gradient_step(data,labels,optimized_theta,layers):
        theta_roll =Multilayer_Perceptron.theta_roll(optimized_theta,layers)
        optimized_theta_roll = Multilayer_Perceptron.feedback_propagation(data,labels,theta_roll,layers)
        optimized_theta_unroll = Multilayer_Perceptron.thetas_unroll(optimized_theta_roll)
        return optimized_theta_unroll  # 梯度向量 

    @staticmethod
    def cost_function(data,labels,thetas,layers):
        """
        计算损失值
        """
        num_layers = len(layers)
        num_samples = data.shape[0]
        num_labels = layers[-1]
        # 前向传播
        predictions = Multilayer_Perceptron.feedforward_propagation(data,thetas,layers)
        # 构建标签矩阵
        bitwise_labels = np.zeros((num_samples,num_labels))
        for sample_index in range (num_samples):
            bitwise_labels[sample_index][labels[sample_index][0]] = 1
        cost1 =np.sum(np.log(predictions[bitwise_labels == 1]))
        cost0 = np.sum(np.log(1- predictions[bitwise_labels == 0]))
        cost = (-1 / num_samples)*(cost1+cost0)
        return cost
    

    @staticmethod
    def feedback_propagation(data,labels,theta_roll,layers):
        """
        Func: 反向传播
        Return: 
        """    
        num_layers = len(layers)
        (num_samples,num_features) = data.shape  # (1700*784)
        num_label_types = layers[-1]
        # 初始化 deltas
        deltas = {}
        for layer_index in range(num_layers-1):
            in_count = layers[layer_index]
            out_count = layers[layer_index+1]
            delta_row = out_count
            delta_col = in_count + 1
            deltas[layer_index] = np.zeros((delta_row,delta_col))   # (25,785)  (10,26)
        

        for sample_index in range(num_samples):
            layers_inputs = {}     # 原始输入
            layers_activations = {}  #  经过激活函数的输入
            # 输入层 默认第0层
            layers_activation = data[sample_index,:].reshape((num_features,1)) # (784,1)
            layers_activations[0] = layers_activation
            
            for layer_index in range(num_layers-1):
                layer_theta = theta_roll[layer_index] # 得到当前权重参数值  (25,785)  (10,26)
                #layer_theta (785,25)  layers_activation(784,1)
                layer_input = np.dot(layer_theta,layers_activation)
                layers_activation = np.vstack((np.array([[1]]),sigmoid(layer_input)))
                layers_inputs[layer_index + 1] = layer_input
                layers_activations[layer_index + 1] = layers_activation
            output_layer_activation = layers_activation[1:,:]
            
            epsilon = {}
            # 标签处理
            bitwise_label = np.zeros((num_label_types,1))
            bitwise_label[labels[sample_index][0]] = 1
            # 计算输出层与真实值的差异
            epsilon[num_layers - 1] = output_layer_activation - bitwise_label
            # 计算每一层与真实值的差值  从后往前 (num_layers - 2, 0,-1)
            for layer_index in range (num_layers - 2,0,-1):
                layer_theta = theta_roll[layer_index] #  theta_t
                next_epsilon = epsilon[layer_index + 1]  #   epsilon_t+1 
                layer_input = layers_inputs[layer_index]  # layer_input_t
                layer_input = np.vstack(((np.array([1])),layer_input))
                # epsilon_t = theta_t.T   *  epsilon_t+1  *  sigmoid_gradient(layer_input_t)
                epsilon_b = np.dot(layer_theta.T,next_epsilon) * sigmoid_gradient(layer_input)
                epsilon[layer_index] =  epsilon_b[1:,:]
            for layer_index in range (num_layers-1):
                layer_delta = np.dot(epsilon[layer_index+1],layers_activations[layer_index].T)
                deltas[layer_index] = deltas[layer_index] + layer_delta
        #  求平均
        for layer_index in range (num_layers-1):
            deltas[layer_index] = deltas[layer_index] * (1/num_samples)

        return deltas

    @staticmethod
    def feedforward_propagation(data,thetas,layers):
        """
        Func: 前向传播
        Return: 输出层结果
        """
        num_samples = data.shape[0]
        num_layers = len(layers)
        in_layer_activation  = data 
        # 逐层处理
        for layer_index in range (num_layers-1):
            theta = thetas[layer_index]
            #  (1700,785) (785,25) ==>(1700,25 + 1) * (26,10) 
            out_layer_activation  = sigmoid(np.dot(in_layer_activation,theta.T))
            out_layer_activation = np.hstack( (np.ones((num_samples, 1)), out_layer_activation))
            in_layer_activation = out_layer_activation
        # 返回输出层结果不需要偏置项
        return in_layer_activation[:,1:]

    @staticmethod
    def  theta_roll(unrolled_thetas,layers):
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
            unrolled_shift = end_index
        return thetas

    @staticmethod
    def thetas_unroll(thetas):
        """
        把所有参数全部排列成一行,有利于参数更新
        """
        num_theta_layers = len(thetas)
        unroll_theta = np.array([])
        for num_layer_index  in range (num_theta_layers):
            # np.hstack 将多个数组沿水平方向（按列）堆叠，等价于 np.concatenate(axis=1)
            unroll_theta = np.hstack((unroll_theta,thetas[num_layer_index].flatten()))
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
            out_count = layers[layer_index+1]
            # out_count,in_count 的顺序是因为是 (6,3)theta * x.T(3,1)
            # 加1 考虑的是偏置项
            thetas[layer_index] = 0.05 * np.random.rand(out_count,in_count+1)
        return thetas