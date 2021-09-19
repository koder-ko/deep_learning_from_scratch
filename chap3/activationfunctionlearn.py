import numpy as np
import matplotlib.pylab as plt


# 활성화 함수 수식
# h(x) = { 0 (x <= 0) / 1 (x > 0)}

# 3.2.2


# 넘파이 배열을 받을 수 없는 step function
# def step_function(x):
#     if x > 0:
#         return 1
#     else:
#         return 0

# 넘파이 배열을 받아도 되는 step function
def step_function(x):
    y = x > 0  # y에 bool 값(true, false) 지정
    return y.astype(np.int64)  # bool 값을 int 로 변환


print(step_function(np.array([-1.0, 1.0, 2.0])))

# 3.2.3

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # ylim: y축 범위 지정
plt.show()


# 3.2.4

# 시그모이드 함수 수식
# h(x) = 1 / 1 + exp(-x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.array([-1.0, 1.0, 2.0])

print(sigmoid(x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()


# 3.2.7

# ReLU 함수 수식
# h(x) = {x (x>0) / 0 (x<=0)}

def relu(x):
    return np.maximum(0, x)  # maximum: 두 입력 중 큰 값을 반환


