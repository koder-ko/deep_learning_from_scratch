import numpy as np
import matplotlib.pylab as plt

# 3.3.2

A = np.array([[1, 2], [3, 4]])
print(A.shape)
B = np.array([[5, 6], [7, 8]])
print(B.shape)
print(np.dot(A, B))  # dot: 행렬의 내적


# 잘못된 행렬 내적
# C = np.array([[1,2], [3,4]])
# print(C.shape)
# print(A.shape)
# print(np.dot(A,C))  에러 발생 이유: 행렬 A의 열 수와 C의 행 수가 같지 않음


# 3.4.3

# 3층 신경망 구현

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identify_function(x):
    return x


from collections import defaultdict


def init_network():
    n = defaultdict()

    # 가중치

    n['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    n['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    n['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])

    n['b1'] = np.array([0.1, 0.2, 0.3])
    n['b2'] = np.array([0.1, 0.2])
    n['b3'] = np.array([0.1, 0.2])

    return n


def forward(n, x):
    W1, W2, W3 = n['W1'], n['W2'], n['W3']
    b1, b2, b3 = n['b1'], n['b2'], n['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identify_function(a3)

    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
