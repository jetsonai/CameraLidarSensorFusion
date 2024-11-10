import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 각도 변환 (도에서 라디안으로)
theta_x = np.deg2rad(30)  # x축으로 30도 회전
theta_y = np.deg2rad(45)  # y축으로 45도 회전
theta_z = np.deg2rad(60)  # z축으로 60도 회전

# 회전 행렬 정의
Rx = np.array([[1, 0, 0],
               [0, np.cos(theta_x), -np.sin(theta_x)],
               [0, np.sin(theta_x), np.cos(theta_x)]])

Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
               [0, 1, 0],
               [-np.sin(theta_y), 0, np.cos(theta_y)]])

Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
               [np.sin(theta_z), np.cos(theta_z), 0],
               [0, 0, 1]])

# 전체 회전 변환 행렬 (Rz * Ry * Rx 순으로 곱함)
rotation_transform = Rz @ Ry @ Rx

# 원래 선분 l의 시작점과 끝점
l_start = np.array([0, 0, 0])
l_end = np.array([2, 2, 0])

# 선분의 시작점과 끝점을 회전 변환
rotated_l_start = rotation_transform @ l_start
rotated_l_end = rotation_transform @ l_end

# 그래프 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 원래 선분 l (녹색)
ax.plot([l_start[0], l_end[0]], [l_start[1], l_end[1]], [l_start[2], l_end[2]], color='green', linewidth=3,  label='Original l')

# 회전된 선분 rotated_l (빨간색)
ax.plot([rotated_l_start[0], rotated_l_end[0]], [rotated_l_start[1], rotated_l_end[1]], [rotated_l_start[2], rotated_l_end[2]], color='red', linewidth=3, label='Rotated l')

# 좌표축 그리기 (검정색)
ax.quiver(0, 0, 0, 1, 0, 0, color='black', length=3)
ax.quiver(0, 0, 0, 0, 1, 0, color='black', length=3)
ax.quiver(0, 0, 0, 0, 0, 1, color='black', length=3)

# 축 범위 설정
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])

# 축 레이블 설정
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# 범례 추가
ax.legend()

# 그래프 보여주기
plt.show()
