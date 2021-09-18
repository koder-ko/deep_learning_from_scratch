import numpy as np

# 1.5.2
x = np.array([1.0, 2.0, 3.0])

print(x)
print(type(x))

# 1.5.3

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print(x + y)
print(x - y)
print(x * y)
print(x / y)

# 1.5.4

A = np.array([[1, 2], [3, 4]])
print(A)
print(A.shape)  # shape는 자료의 형상을 나타냄 ex) 2 x 3 배열
print(A.dtype)  # dtype은 원소의 자료형을 나타냄 ex) int 64

print(A * 10)  # 브로드캐스트

# 넘파이 배열은 N차원 배열을 작성할 수 있는데. 1차원 배열을 수학에서 벡터 2차원을 행렬 벡터와 행렬의 일반화가 텐서이다.
# 책대로 2차원 배열은 행렬, 3차원 이상의 배열을 다차원으로 작성함.

# 1.5.5

A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
print(A * B)  # 브로드캐스트: 형상이 다른 배열 간의 곱을 확대시켜 연산하는 것.

# 1.5.6

X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
print(X[0])  # 인덱스
print(X[0][1])

for row in X:
    print(row)

X = X.flatten()  # X를 1차원 배열로 변환(평탄화)
print(X)

print(X[np.array([0, 2, 4])])  # 인덱스가 0,2,4 인 원소를 인덱스 숫자의 배열로 획득.

print(X[X > 15])  # 15이상인 값만 출력

