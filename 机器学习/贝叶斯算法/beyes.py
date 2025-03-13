import numpy as np
import re
import random

from sklearn import tests
import test
"""
email文件夹下有两个文件夹ham和spam。ham文件夹下的txt文件为正常邮件 spam文件下的txt文件为垃圾邮件。

1收集数据:提供文本文件。
2准备数据:将文本文件解析成词条向量。
3分析数据:检查词条确保解析的正确性。
4训练算法:计算不同的独立特征的条件概率。
5测试算法:计算错误率。
6使用算法:构建一个完整的程序对一组文档进行分类

"""

# 将邮件转化为字符串列表
def textPrase(input_string):
    listofTokens = re.split(r'\W+' , input_string)
    # 把所有单词转化为小写，单个大写字母不处理
    return [tok.lower() for tok in listofTokens  if len(tok) > 2]

# 创建语料表
def createVocalist(doclist):
    vocabSet = set([])
    for document in doclist:
        #  | 取并集 ， & 取交集
        vocabSet = vocabSet|set(document)
    return list(vocabSet)

# 创建文本向量
def setOfWord2Vec(Vocalist,inputset):  
    returnVec = [0] * len(Vocalist)
    for word in inputset:
        if word in Vocalist:
            returnVec[Vocalist.index(word)] = 1
    return returnVec


# 统计先验概率
def trianNB(trainMat,trainClass):
    """
        trainMat: 邮件向量
        trainClass: 邮件类别
    """
    num_trian = len(trainMat)  # 训练样本个数 
    num_words = len(trainMat[0])  # 邮件特征个数
    # 垃圾邮件概率值 p(spam) = 垃圾邮件个数/邮件总数
    p_spam = sum(trainClass) / float(num_trian)
    
    """拉普拉斯平滑
        词出现次数 初始化为1 拉普拉斯平滑处理 防止概率为0
        分子 ：这里平滑系数取 1   分母: n * 1  n是特征总数 这里取 2 代表的是类别总数 
        P(xi|Y) = (count(xi,Y) + a) / (count(Y) + a*n)
        例 p(meeting|spam)
    """
    xi_ham_cnt = np.ones((num_words)) 
    xi_spam_cnt = np.ones((num_words)) 
    ham_cnt = 2
    spam_cnt = 2
    
    for i in range(num_trian):
        if trainClass[i] == 1: #spam
            xi_spam_cnt += trainMat[i]  # 这里是一个数组 存放每个单词出现的次数  即词频
            spam_cnt += sum(trainMat[i]) # 这是一个数 记录所有spam中的次的个数
        else:
            xi_ham_cnt += trainMat[i]
            ham_cnt += sum(trainMat[i])

    # 取log 放大概率结果
    p_xi_spam = np.log(xi_spam_cnt / spam_cnt)
    p_xi_ham = np.log(xi_ham_cnt / ham_cnt)
    return p_xi_ham,p_xi_spam,p_spam


# 预测关键步骤 log处理 如何推到公式？
def classifyNB(wordvec, p_xi_ham,p_xi_spam,p_spam):
    p_spam_xi = np.log(p_spam) + sum(wordvec*p_xi_spam)
    p_ham_xi = np.log(1-p_spam) + sum(wordvec*p_xi_ham)
    if p_spam_xi > p_ham_xi:
        return 1
    else:
        return 0
    

def spam():
    doclist = []
    classlist = [] 
    for i in range(1,26):
        wordlist = textPrase(open('./email/spam/%d.txt'% i, 'r').read())
        doclist.append(wordlist)
        classlist.append(1) # 1 表示垃圾邮件

        wordlist = textPrase(open('./email/ham/%d.txt'% i, 'r').read())
        doclist.append(wordlist)
        classlist.append(0) # 0 表示正常邮件
    
    # 创建语料表
    Vocalist = createVocalist(doclist)

    # 数据集划分
    trianset = list(range(50))
    testset = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trianset)))
        testset.append(trianset[randIndex])
        del trianset[randIndex]

    # 训练
    trainMat = []
    trainClass = []
    for docIndex in trianset:
        trainMat.append(setOfWord2Vec(Vocalist,doclist[docIndex]))
        trainClass.append(classlist[docIndex])

    p_xi_ham,p_xi_spam,p_spam = trianNB(np.array(trainMat),np.array((trainClass)))


    error_cnt = 0
    for docIndex in testset:
        wordvec = setOfWord2Vec(Vocalist,doclist[docIndex])
        res = classifyNB(np.array(wordvec) , p_xi_ham,p_xi_spam,p_spam)
        if  res != classlist[docIndex]:
            error_cnt += 1
    print(f"error clf : {error_cnt}")



spam()
