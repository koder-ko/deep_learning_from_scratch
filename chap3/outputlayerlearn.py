import numpy as np


# 1. 항등 함수
# a1 = y1 a2 = y2 a3 = y3


# 2. 소프트맥스 함수
# yk = exp(ak) / n sig i=1 exp(ai)

# 3.5.1

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


a = np.array([0.3, 2.9, 4.0])

y = softmax(a)

print(y)


# softmax overflow problem: 지수함수라 수치가 매우 커지고 나눗셈시 수치가 불안정 해짐
# softmax 의 변형
# yk = exp(ak + C') / n sig i=1 exp(ai + C')


# 3.5.2

def bettersoftmax(a):
    c = np.max(a)  # max: 입력된 배열 중에서 가장 큰 값 반환
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


a = np.array([1010, 1000, 900])

# print(softmax(a))  에러 발생 이유: 값이 너무 커서 overflow 에러 발생

print(bettersoftmax(a))


# 3.5.3

a = np.array([0.3, 2.9, 4.0])
y = bettersoftmax(a)
print(y)
print(np.sum(y))  # softmax 출력 총합은 1







