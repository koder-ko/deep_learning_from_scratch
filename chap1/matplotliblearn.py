import numpy as np
import matplotlib.pyplot as plt  # pyplot 모듈은 그래프를 그리는데 사용됨

# 1.6.1

x = np.arange(0, 6, 0.1)  # 0에서 6까지 0.1 간격으로 생성
y = np.sin(x)

plt.plot(x, y)  # plot: 그래프 생성
plt.show()

# 1.6.2

x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin")  # label: 그래프에 라벨 붙임
plt.plot(x, y2, linestyle="--", label="cos")  # linestyle: 그래프 선의 모습

# linestyle에 사용할 수 있는 기호들
# '-'  : solid
# '--' : dashed
# '-.' : dashdot
# ':'  : dotted


plt.xlabel("x")  # xlabel: x축의 라벨
plt.ylabel("y")  # y축의 ylabel: y축의 라벨
plt.title("sin & cos")  # title: 그래프 타이틀
plt.legend()  # legend: 범례 표시
plt.show()

# 1.6.3

from matplotlib.image import imread


img = imread('lena.png')  # imread: 이미지 리딩

plt.imshow(img)  # imshow: 이미지 렌더
plt.show()
