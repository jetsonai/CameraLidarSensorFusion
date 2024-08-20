import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 초기 점 P의 좌표
P = np.array([1, 1, 1])

# 변환 행렬 (예: 이동 변환)
translation_matrix = np.array([[2, 0, 0], [0, -1, 0], [0, 0, 3]])

# 점 P를 변환 행렬과 곱해 변환된 점 계산
translated_P = np.dot(translation_matrix, P)

# 변환된 점 출력
print("Initial Point P:", P)
print("Translated Point:", translated_P)

# 그래프 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 좌표축 그리기
ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=3)
ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=3)
ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=3)

# 초기 점 P 표시 (녹색)
ax.scatter(P[0], P[1], P[2], color='green', s=100, label='Initial P')

# 변환된 점 translated_P 표시 (빨간색)
ax.scatter(translated_P[0], translated_P[1], translated_P[2], color='red', s=100, label='Translated P')

# 축 범위 설정
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])

# 축 레이블 설정
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# 범례 추가
ax.legend()

# 그래프 보여주기
plt.show()
