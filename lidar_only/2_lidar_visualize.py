import open3d as o3d
import os

# 지정된 디렉토리 경로
pcd_file = "/home/ubuntu/Desktop/sensor_fusion/lidar_only/pcd/0000000000.pcd"

# .pcd 파일 읽기
pcd = o3d.io.read_point_cloud(pcd_file)

# 파일 이름 출력
print(f"Visualizing: {pcd_file}")

# Open3D로 시각화
o3d.visualization.draw_geometries([pcd])
