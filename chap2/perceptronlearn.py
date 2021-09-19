import numpy as np


# 퍼셉트론 수식 - 1
# y = {0 (w1x1 + w1x2 <= theta) / 1 (w1x1 + w1x2 > theta)}
# x1, x2 : 입력신호
# w1, w2 : 가중치

# AND 게이트 진리표
# 0 0 -> 0
# 1 0 -> 0
# 0 1 -> 0
# 1 1 -> 1


# NAND 게이트 진리표
# 0 0 -> 1
# 1 0 -> 1
# 0 1 -> 1
# 1 1 -> 0

# OR 게이트 진리표
# 0 0 -> 0
# 1 0 -> 1
# 0 1 -> 1
# 1 1 -> 0


# 2.3.1

# 단순한 구현

def simpleAND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


print(simpleAND(0, 0))
print(simpleAND(0, 1))
print(simpleAND(1, 0))
print(simpleAND(1, 1))


# 퍼셉트론 수식 - 2
# y = {0 (b + w1x1 + w1x2 <= 0) / 1 (b + w1x1 + w1x2 > 0)}
# x1, x2 : 입력신호
# w1, w2 : 가중치
# b      : 편향 => -theta

# 2.3.2

# 편향 도입

def biasAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


print(biasAND(0, 0))
print(biasAND(0, 1))
print(biasAND(1, 0))
print(biasAND(1, 1))


def biasNAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


print(biasNAND(0, 0))
print(biasNAND(0, 1))
print(biasNAND(1, 0))
print(biasNAND(1, 1))


def biasOR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


print(biasOR(0, 0))
print(biasOR(0, 1))
print(biasOR(1, 0))
print(biasOR(1, 1))


# XOR 다층퍼셉스론 구현시 1층: NAND, OR  2층: AND
# XOR 게이트 진리표
# x1 x2   s1 s2   y
# 0  0 -> 1  0 -> 0
# 1  0 -> 1  1 -> 1
# 0  1 -> 1  1 -> 1
# 1  1 -> 0  1 -> 0

# 2.5.2

def biasXOR(x1, x2):
    s1 = biasNAND(x1, x2)
    s2 = biasOR(x1, x2)
    y = biasAND(s1, s2)
    return y


print(biasXOR(0, 0))
print(biasXOR(0, 1))
print(biasXOR(1, 0))
print(biasXOR(1, 1))


