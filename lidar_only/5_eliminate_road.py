import open3d as o3d
import numpy as np

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
# 평면 모델: ax + by + cz + d = 0
plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1,
                                         ransac_n=3,
                                         num_iterations=2000)

[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# 도로에 속하는 포인트 (inliers)
road_pcd = ror_pcd.select_by_index(inliers)

# 도로에 속하지 않는 포인트 (outliers)
non_road_pcd = ror_pcd.select_by_index(inliers, invert=True)

# 도로가 아닌 포인트만 추출하여 시각화 (여기에 물체가 있을 가능성이 있음)
non_road_pcd.paint_uniform_color([0, 1, 0])  # 빨간색으로 표시

# 도로 영역을 시각화 (도로는 초록색으로 표시)
road_pcd.paint_uniform_color([1, 0, 0])

# 두 영역을 동시에 시각화
o3d.visualization.draw_geometries([road_pcd, non_road_pcd], 
                                  window_name="Road (Red) and Non-Road (Green) Points")

o3d.visualization.draw_geometries([non_road_pcd], window_name="Final PCD")
