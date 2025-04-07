# 一个项集是非频繁集，那么它的所有超级也是非频繁的
# 通过频繁项集生成关联规则。
# 采用逐层搜索的方法，从小的项集开始，不断扩展，直到找到所有满足最低支持度和置信度的规则。

from re import sub


def loadDataset():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1 (dataset):
    C1 = []  # 一项集
    for transaction in dataset:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # 把C1当中每个元素转化为集合的元素
    return list(map(frozenset,C1))

def scanD(D,CK,min_support):
    """
    Args:
        D : 数据集
        CK : K项集
        min_support : 最小支持度
    """
    sscant = {}  # 统计每个项集出现的次数
    for tid in D:
        for can in CK:
            # 判断是否是子集
            if can.issubset(tid):
                if not can in  sscant:
                    sscant[can] =1 
                else:
                    sscant[can] +=1 
    numItems = float(len(list(D)))

    retlist = []
    supportData = {}
    for key in sscant:
        support = sscant[key] / numItems 
        if support >= min_support:
            retlist.insert(0,key)
        supportData[key] = support
    return retlist ,supportData


# 拼接操作
def apriorigen(LK,k):
    retlist= [ ]
    lenLK = len(LK)
    for i in range (lenLK):
        for j in range(i+1,lenLK):
            L1 = list(LK[i])[:k-2]
            L2 = list(LK[j])[:k-2]
            if L1 == L2:
                retlist.append(LK[i]|LK[j])
    return retlist

def apriori(dataset , min_support = 0.5):
    c1 = createC1(dataset) # 创建一项集 
    L1 ,supportData = scanD( dataset ,c1 , min_support) # 过滤掉支持度不满足的项集
    L = [L1]
    k = 2 
    while(len(L[k-2]) > 0):
        CK = apriorigen(L[k-2] , k)
        LK , support_k = scanD( dataset ,CK , min_support)
        supportData.update(support_k)
        L.append(LK)
        k += 1
    return L, supportData


def generateRules(L , supportData , min_conf=0.6 ):
    rulelist = []
    for i in range(1,len(L)):
        for freqset in L[i]:
            # frozenset 不可变的集合 ,创建后无法修改（增删元素）
            H1 = [frozenset([item]) for item in freqset]  # [set(1) , set(2)]
            rulesFromConseq(freqset ,H1 ,supportData ,rulelist,min_conf)


def rulesFromConseq(freqset ,H,supportData ,rulelist,min_conf):
    m = len(H[0])
    while(len(freqset) > m ):  # ?
        H = calConf(freqset , H,supportData ,rulelist,min_conf)
        if(len(H)>1):
            apriorigen(H,m+1)
            m+=1
        else:
            break

def calConf(freqset ,H, supportData ,rulelist,min_conf):
    prunedh = []
    for conseq in H:
        conf = supportData[freqset] / supportData[freqset - conseq ]
        if  conf >= min_conf:
            print(freqset-conseq,'-->',conseq, f'conf = {conf}')
            rulelist.append((freqset-conseq,conseq,conf))
            prunedh.append(conseq)
    return prunedh


"""
if __name__ == "__main__":
区分当前脚本是直接被运行，还是被作为模块导入到其他代码中
__name__ 是 Python 文件的内置属性：
当文件直接运行时__name__ 的值为 "__main__"。
当文件被导入为模块时__name__ 的值为模块名
"""
if __name__ == "__main__":
    dataset = loadDataset()
    L ,support = apriori(dataset)
    i = 0
    for freq in L:
        print('项数：',i+1 , freq)
        i+=1
    rules = generateRules(L,support,min_conf=0.5)