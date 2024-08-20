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

# DBSCAN 클러스터링 적용
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(final_point.cluster_dbscan(eps=0.6, min_points=11, print_progress=True))

# 각 클러스터를 색으로 표시
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

# 노이즈를 제거하고 각 클러스터에 색상 지정
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0  # 노이즈는 검정색으로 표시
final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])

# 바운딩 박스 생성 및 시각화
bboxes = []
for i in range(max_label + 1):
    cluster_pcd = final_point.select_by_index(np.where(labels == i)[0])
    bbox = cluster_pcd.get_axis_aligned_bounding_box()
    bbox.color = (1, 0, 0)  # 바운딩 박스 색상 (빨강)
    bboxes.append(bbox)
    
    bbox_coordinate_min = bboxes[i].min_bound
    bbox_coordinate_max = bboxes[i].max_bound
    print("Object BBox Coordinate : ", bbox_coordinate_min, bbox_coordinate_max)

# 시각화: 클러스터와 바운딩 박스를 함께 표시
o3d.visualization.draw_geometries([final_point] + bboxes, 
                                  window_name="DBSCAN Clustered Points with Bounding Boxes")
