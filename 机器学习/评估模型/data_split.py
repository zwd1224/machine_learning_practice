import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import warnings
import scipy.io as sio
import warnings
warnings.filterwarnings('ignore')
# 70000张  28*28*1
mnist = sio.loadmat('mnist-original.mat')

# 数据切分  6:1
X,y = mnist["data"].T, mnist["label"].T.flatten()
# print(X.shape)
# print(y.shape)

y = y.astype(np.uint8) 
X_train, X_test, y_train,y_test = X[:60000] , X[60000:],y[:60000] , y[60000:]

# 打乱顺序 ，排除顺序的影响，洗牌操作
shuffle_index = np.random.permutation(60000)
X_train,y_train = X_train[shuffle_index],y_train[shuffle_index]


# K折交叉验证
k = 3
# 新标签  0 、1  是5  不是5
y_train_5 = (y_train==5)
y_test_5 = (y_test==5)

from sklearn.linear_model import SGDClassifier  
from sklearn.model_selection import cross_val_score

# SDG分类器
sgd_clf = SGDClassifier(max_iter=5,random_state= 42)
# random_state=42 是数据科学和机器学习领域中经常见到的一个惯例。
# 任何整数都可以用作随机种子，而选择 42 只是开发者的一种习惯。

# fit  fit() 方法是许多机器学习模型的核心部分，在训练阶段用于拟合数据并调整模型参数。
# 通过传递特征矩阵 X 和目标变量 y 给此方法，可以使得模型能够自动找到最佳权重来最小化预测误差。
sgd_clf.fit(X_train,y_train_5) 
# 预测单个样本
print(sgd_clf.predict([X[35000]]))  # [ True]
print(y[35000]) # 5 

# 交叉验证API
#分类器   训练集    标签   K折   精度
accuracy1 = cross_val_score(sgd_clf, X_train , y_train_5 , cv = k, scoring = 'accuracy' )
# print(accuracy1)  

# 交叉验证手动实现
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone 
skflods = StratifiedKFold(n_splits = 3 , shuffle=True, random_state = 42)
#  6: 1    6 => 2 ：1  再在测试集里进行切分3段 ，2段为训练，1段为测试
for train_index , test_index in skflods.split(X_train,y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_flods = X_train[train_index]
    y_trian_flods = y_train_5[train_index]
    X_test_flods = X_train[test_index]
    y_test_flods = y_train_5[test_index]
    
    clone_clf.fit(X_train_flods,y_trian_flods)
    y_pred = clone_clf.predict( X_test_flods)
    n_correct = sum(y_pred == y_test_flods )
    # print(n_correct / len(y_pred))


# 混淆矩阵
# TP: 正例预测为正  FP： 负例预测为正  FN：正例预测为负  TN：负例预测为负
# 召回率：recall = TP / (TP + FN)
# 精度: accuracy  = TP / (TP + FP)
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score,f1_score

# 获取交叉验证预测值
y_train_pred = cross_val_predict(sgd_clf,X_train , y_train_5,cv = 3)
confusion_matrix(y_train_5,y_train_pred)
# [[53466(TN)  1113(FP)]
#  [ 1229(FN)  4192(TP)]]

precision_score(y_train_5,y_train_pred) # 0.7463463799764825
recall_score(y_train_5,y_train_pred)# 0.819590481460985

# F1 score  = 2(Accuracy·Racall)/(Accuracy+Racall) ————调和平均数
f1_score(y_train_5,y_train_pred)

# 阈值 模型计算结果  
# 获得模型的计算值，可以自己设计阈值判断结果
sgd_clf.decision_function([X[35000]]) # [177465.45082867]

from sklearn.metrics import precision_recall_curve
# 获得每个样本的模型计算值，而不是最终输出结果
y_scores = cross_val_predict(sgd_clf,X_train , y_train_5,cv = 3,method='decision_function')

# thresholds 阈值
# 用于绘制分类模型在不同阈值下的 精确率 (Precision) 和 召回率 (Recall) 的变化曲线。
# 它常用于评估模型在不平衡数据集上的表现，特别是当正负样本的比例悬殊时。
precisions,recalls,thresholds = precision_recall_curve(y_train_5,y_scores)

# ROC曲线 : true positive rate(TPR)  AND  false positive rate(FPR)
# TPR = TP / (TP + FN) -- Recall  越大越好
# FPR = FP / (FP + TN)   越小越好
