# pip install hdbscan

import open3d as o3d
import numpy as np
import hdbscan
import matplotlib.pyplot as plt

# PCD 파일 경로 설정
pcd_path = "/home/ubuntu/Desktop/sensor_fusion/lidar_only/pcd/0000000001.pcd"

# PCD 파일 읽기
original_pcd = o3d.io.read_point_cloud(pcd_path)

# Voxel Downsampling 수행
voxel_size = 0.4  # 필요에 따라 voxel 크기를 조정하세요.
downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

# Radius Outlier Removal (ROR) 적용
cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
ror_pcd = downsample_pcd.select_by_index(ind)

# RANSAC을 사용하여 평면 추정
plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1,
                                             ransac_n=3,
                                             num_iterations=2000)

# 도로에 속하지 않는 포인트 (outliers) 추출
final_point = ror_pcd.select_by_index(inliers, invert=True)

# 포인트 클라우드를 NumPy 배열로 변환
points = np.asarray(final_point.points)

# HDBSCAN 클러스터링 적용
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=1)
labels = clusterer.fit_predict(points)

# 각 클러스터를 색으로 표시
max_label = labels.max()
print(f"Point cloud has {max_label + 1} clusters")

# 노이즈를 제거하고 각 클러스터에 색상 지정
colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
colors[labels < 0] = 0  # 노이즈는 검정색으로 표시
final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])

# 시각화
o3d.visualization.draw_geometries([final_point], 
                                  window_name="HDBSCAN Clustered Points")
