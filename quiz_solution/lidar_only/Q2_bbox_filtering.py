import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# PCD 파일 경로 설정
pcd_path = "/home/ubuntu/Desktop/sensor_fusion/lidar_only/pcd/0000000001.pcd"

# PCD 파일 읽기
original_pcd = o3d.io.read_point_cloud(pcd_path)

# Voxel Downsampling 수행
voxel_size = 0.5  # 필요에 따라 voxel 크기를 조정하세요.
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

# DBSCAN 클러스터링 적용
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(final_point.cluster_dbscan(eps=0.6, min_points=11, print_progress=True))

# 각 클러스터를 색으로 표시
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

# 노이즈를 제거하고 각 클러스터에 색상 지정
colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
colors[labels < 0] = 0  # 노이즈는 검정색으로 표시
final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])

# 필터링 기준
min_points_in_cluster = 15   # 클러스터 내 최소 포인트 수
max_points_in_cluster = 40  # 클러스터 내 최대 포인트 수

# 1번 조건을 만족하는 클러스터 필터링 및 바운딩 박스 생성
bboxes_1 = []
for i in range(max_label + 1):
    cluster_indices = np.where(labels == i)[0]
    if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
        cluster_pcd = final_point.select_by_index(cluster_indices)
        
        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        bbox.color = (1, 0, 0)  # 바운딩 박스 색상 (빨강)
        bboxes_1.append(bbox)

# 필터링 기준 추가
min_z_value = -2.0    # 클러스터 내 최소 Z값
max_z_value = 0.5   # 클러스터 내 최대 Z값

# 1번, 2번 조건을 만족하는 클러스터 필터링 및 바운딩 박스 생성
bboxes_12 = []
for i in range(max_label + 1):
    cluster_indices = np.where(labels == i)[0]
    if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
        cluster_pcd = final_point.select_by_index(cluster_indices)
        z_values = np.asarray(cluster_pcd.points)[:, 2]  # Z값 추출
        print("z_min : ", z_values.min(), "  z-max : ", z_values.max() )
        if min_z_value <= z_values.min() and z_values.max() <= max_z_value:
            bbox = cluster_pcd.get_axis_aligned_bounding_box()
            bbox.color = (1, 0, 0)  # 바운딩 박스 색상 (초록)
            bboxes_12.append(bbox)

# 시각화: 1번, 2번 조건을 만족하는 클러스터와 바운딩 박스
o3d.visualization.draw_geometries([final_point] + bboxes_12, 
                                  window_name="Clusters filtered by Condition 1, 2")
