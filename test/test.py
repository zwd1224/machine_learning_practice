import numpy as np

# 2阶单位矩阵
E = np.identity(2 , dtype=int)
# 对角矩阵
D = np.diag([1,2,3])
# 1 矩阵和0矩阵
Z = np.zeros((2,2))
O = np.ones((2,2))

A = np.arange(9).reshape((3,3))
# 矩阵转置
B = A.T
# 矩阵加法
C = A + B
# 矩阵减法
C = A - B
# 矩阵乘法
C = A @ B

# 向 量 点 乘
a = np.array([3, 4, 6])
b = np.array([5, 2, 8])
c = a @ b
print(c)
