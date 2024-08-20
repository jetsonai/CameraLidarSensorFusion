import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 원래 점들
P1 = np.array([1, 2, 3])
P2 = np.array([1, 3, 2])
P3 = np.array([3, 1, 2])

# 점들을 배열로 묶기
triangle = np.array([P1, P2, P3, P1])

# 투영 행렬 정의
projection_xy = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]])

projection_xz = np.array([[1, 0, 0],
                          [0, 0, 0],
                          [0, 0, 1]])

projection_yz = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

# 각 평면에 투영된 점들 계산
xy_triangle = np.dot(triangle, projection_xy.T)
xz_triangle = np.dot(triangle, projection_xz.T)
yz_triangle = np.dot(triangle, projection_yz.T)

# 그래프 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 원래 삼각형 (검정색)
ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], color='black', label='Original Triangle')

# xy 평면에 투영된 삼각형 (빨간색)
ax.plot(xy_triangle[:, 0], xy_triangle[:, 1], xy_triangle[:, 2], color='red', label='XY Projection')

# xz 평면에 투영된 삼각형 (녹색)
ax.plot(xz_triangle[:, 0], xz_triangle[:, 1], xz_triangle[:, 2], color='green', label='XZ Projection')

# yz 평면에 투영된 삼각형 (파란색)
ax.plot(yz_triangle[:, 0], yz_triangle[:, 1], yz_triangle[:, 2], color='blue', label='YZ Projection')

# 좌표축 그리기 (검정색)
ax.quiver(0, 0, 0, 1, 0, 0, color='black', length=4)
ax.quiver(0, 0, 0, 0, 1, 0, color='black', length=4)
ax.quiver(0, 0, 0, 0, 0, 1, color='black', length=4)

# 축 범위 설정
ax.set_xlim([0, 4])
ax.set_ylim([0, 4])
ax.set_zlim([0, 4])

# 축 레이블 설정
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# 범례 추가
ax.legend()

# 그래프 보여주기
plt.show()
