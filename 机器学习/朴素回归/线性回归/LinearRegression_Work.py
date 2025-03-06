from LinearRegression import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('world-happiness-report-2017.csv')
train_data = data.sample(frac = 0.8)
test_data = data.drop(train_data.index)
input_param_name1 = 'Economy..GDP.per.Capita.'
input_param_name2 = 'Freedom'
output_param_name = 'Happiness.Score'

x_train = train_data[[input_param_name1]].values
y_train = train_data[[output_param_name]].values
x_test = test_data[[input_param_name1]].values
y_test = test_data[[output_param_name]].values

# plt.scatter(x_train,y_train)
# plt.scatter(x_test,y_test)
# plt.show()

lr = LinearRegression(x_train, y_train)
alpha = 0.01 
num_iterations = 500
(theta,cost_history) = lr.train(alpha , num_iterations)
print(cost_history[0])
print(cost_history[-1])
# plt.plot(range(len(cost_history)), cost_history)
# plt.show()


predictions_num = 100
x_pre = np.linspace(x_train.min(), x_train.max(),predictions_num).reshape(predictions_num,1)
y_pre = lr.get_predict(x_pre)
plt.scatter(x_train,y_train)
plt.scatter(x_test,y_test)
plt.plot(x_pre,y_pre,'r')
plt.show()