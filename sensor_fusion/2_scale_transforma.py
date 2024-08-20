import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 원점에서 반지름 2인 구 생성
def create_sphere(radius, center, num_points=100):
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    return x, y, z

# 크기 변환 행렬
scale_matrix = np.array([[0.3, 0, 0], 
                         [0, -0.3, 0], 
                         [0, 0, 3]])

# 원점 중심의 구
center = np.array([0, 0, 0])
radius = 2
x_s, y_s, z_s = create_sphere(radius, center)

# 구의 좌표들을 변환 행렬과 곱하여 변환된 구 생성
x_scaled_s = scale_matrix[0, 0] * x_s + scale_matrix[0, 1] * y_s + scale_matrix[0, 2] * z_s
y_scaled_s = scale_matrix[1, 0] * x_s + scale_matrix[1, 1] * y_s + scale_matrix[1, 2] * z_s
z_scaled_s = scale_matrix[2, 0] * x_s + scale_matrix[2, 1] * y_s + scale_matrix[2, 2] * z_s

# 그래프 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 초기 구 (녹색)
ax.plot_surface(x_s, y_s, z_s, color='green', alpha=0.6)

# 변환된 구 (빨간색)
ax.plot_surface(x_scaled_s, y_scaled_s, z_scaled_s, color='red', alpha=0.6)

# 좌표축 그리기
ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=4)
ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=4)
ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=4)

# 축 범위 설정
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 10])

# 축 레이블 설정
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# 그래프 보여주기
plt.show()
