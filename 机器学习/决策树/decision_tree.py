import math
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def createDataSet():
    dataSet = [
        [0, 0, 0, 0, 'no'],
        [0, 0, 0, 1, 'no'],
        [0, 1, 0, 1, 'yes'],
        [0, 1, 1, 0, 'yes'],
        [0, 0, 0, 0, 'no'],
        [1, 0, 0, 0, 'no'],
        [1, 0, 0, 1, 'no'],
        [1, 1, 1, 1, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [2, 0, 1, 2, 'yes'],
        [2, 0, 1, 1, 'yes'],
        [2, 1, 0, 1, 'yes'],
        [2, 1, 0, 2, 'yes'],
        [2, 0, 0, 0, 'no']
    ]
    labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']
    return dataSet, labels

def createTree(data , labels , feature_lables):
    
    # 递归的结束条件：1. 到达叶子节点 2. 所有特征都已添加
    # 如果当前节点的结果唯一 都是no或者都是yes
    classList =[example[-1] for example in data]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果所有样本都选择完，则返回
    if len(data[0]) == 1 :
        return majorityCnt(classList)
    
    # 创建当前节点
    # 得到最优特征的下标
    best_feature = choose_best_feature(data)
    # 获得最优特征的名字
    best_feature_label = labels[best_feature]
    # 添加最优特征，记录树的特征顺序！
    feature_lables.append(best_feature_label)
    myTree = {best_feature_label:{}}
    # 创建一个节点，删除数据中对应特征 
    del labels[best_feature]  ##### best_feature
    sublabels = labels[:]

    # 创建子节点,计算分支个数
    feature_values = [example[best_feature] for example in data]
    unique_values = set(feature_values)
    for value in unique_values:
        # 调用递归函数
        myTree[best_feature_label][value] = createTree(splitDate(data,best_feature,value),sublabels,feature_lables)
    return myTree


# 计算节点中，那种结果个数多
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # key=operator.itemgetter(1) 返回元组中的第二个元素,让 sorted() 函数根据字典的值（即计数）进行排序
    sortedclassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedclassCount[0][0]

def choose_best_feature(data):
    num_feature = len(data[0]) - 1  # 特征总个数
    baseEntropy = cal_shannon_Ent(data)  # 起始的熵值
    bestInfoGain = 0   # 记录最大信息增益
    best_feature = -1  # 记录信息增益最大的特征下标
    for i in range(num_feature):
        # 获得某个特征的所有数据
        feature = [example[i] for example in data]
        unique_values = set(feature)  # 每个特征值
        ent = 0
        for val in unique_values:
            subdata = splitDate(data , i , val)
            prob = len(subdata)/float(len(data))
            # 天气信息增益 = (雨天个数/所有天) * 雨天熵 + (晴天个数/所有天) * 晴天熵 + (阴天个数/所有天) * 阴天熵
            ent += prob * cal_shannon_Ent(subdata)
        infogain = baseEntropy - ent
        if infogain > bestInfoGain:
            bestInfoGain = infogain
            best_feature = i
    return best_feature

# 计算某一个特征值（天气： 雨天）的熵值,雨天的总个数，正例个数，负例个数
def cal_shannon_Ent(data):
    num_sample = len(data)
    label_counts = {}
    for sample in data:
        current_label = sample[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1    
    ent = 0
    # 熵值 = - 雨天打球/所有雨天 * log（雨天打球/所有雨天）- 雨天不打球/所有雨天 * log（雨天不打球/所有雨天）
    for key in label_counts:
        prop = float(label_counts[key])/num_sample
        ent -= prop*math.log2(prop)
    return ent
# 去掉原数据中的某一列
def splitDate(data,axis,value):
    subdata = []
    for featvac in data:
        if featvac[axis] == value:
            tempdata = featvac[:axis]
            tempdata.extend(featvac[axis:])
            subdata.append(tempdata)
    return subdata


"""
当模块作为主程序运行时：
__name__ 的值会被设置为 '__main__'。
if __name__ == '__main__': 条件成立，其下的代码块会被执行。

当模块被其他模块导入时：
__name__ 的值会被设置为模块的名字（即文件名，去掉 .py 后缀）。
if __name__ == '__main__': 条件不成立，其下的代码块不会被执行。
"""

if __name__ == '__main__':
    data,labels = createDataSet()
    featlabels = []
    MyTree = createTree(data,labels,featlabels)
    print(MyTree)
    # {'F3-HOME': {0: {'F2-WORK': {0: 'no', 1: 'yes'}}, 1: 'yes'}}