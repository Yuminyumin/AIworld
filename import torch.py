import numpy as np # 넘파이 사용
import matplotlib.pyplot as plt # 맷플롯립사용

def sigmoid(x): # 시그모이드 함수 정의
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y, 'g') #g는 색상
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가, [0,0],[1.0,0.0] = x축과 y축의 값을 지정 :은 점선 스타일
plt.title('Sigmoid Function')
plt.show()