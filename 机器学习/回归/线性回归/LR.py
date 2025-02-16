# 线性回归
# 权重项  偏置项   优化函数 
import numpy as np
from utils.features.prepare_for_training import prepare_for_training
class LinearRegression:
   
    def __init__(self, data, labels , polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        polynomial_degree = 0 不懂
        sinusoid_degree = 0  不懂
        normalize_data 标准化数据
        features_mean 标准化后的均值
        features_deviation 标准化后的方差
        """
        (data_processed, 
         features_mean, 
         features_deviation) = prepare_for_training(data,polynomial_degree=0, sinusoid_degree=0, normalize_data=True)
        
        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features,1))  # theta竖着的,data是横着的 ==>np.dot(data ,theta)

    def  train(self , alpha ,num_iterations = 500):
        cost_history =  self.gradient_descent(alpha,num_iterations)
        return self.theta,cost_history
    def gradient_descent(self, alpha, num_iterations):
        """
        循环迭代,跟新theta
        """
        cost_history= []
        for i in range( num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data,self.labels))
        return cost_history
    def gradient_step(self,alpha):
        """
        梯度下降核心
        """
        num_examples = self.data.shape[0]
        prediction = np.dot(self.data ,self.theta)
        delta = prediction - self.labels  #  y[i] -theta[i]*x[i]   预测值减去真实值
        theta = self.theta
        # 小批量梯度下降，theta更新方法
        theta = theta - alpha*(1/num_examples)*(np.dot(delta.T,self.data)).T
        self.theta = theta 
    def cost_function(self,data,labels):
        num_examples = self.data.shape[0]
        prediction = np.dot(data ,self.theta)
        delta = prediction - labels  #  y[i] -theta[i]*x[i]   预测值减去真实值
        cost = (1/2)*np.dot(delta.T , delta) / num_examples
        return cost[0][0]
    def get_cost(self, data , labels):
        return self.cost_function(data,labels)
    
    def get_predict(self, data):
        data_processed = prepare_for_training(data,polynomial_degree=0, sinusoid_degree=0, normalize_data=True)[0]
        predictions =  np.dot(data_processed ,self.theta)
        return predictions