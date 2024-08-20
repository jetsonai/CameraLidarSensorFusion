import open3d as o3d

# PCD 파일 경로 설정
pcd_path = "/home/ubuntu/Desktop/sensor_fusion/lidar_only/pcd/0000000001.pcd"

# PCD 파일 읽기
original_pcd = o3d.io.read_point_cloud(pcd_path)

# 원본 포인트 개수 출력
print(f"Original point count: {len(original_pcd.points)}")

# Voxel Downsampling 수행
voxel_size = 0.5  # 필요에 따라 voxel 크기를 조정하세요.
downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

# 다운샘플링된 포인트 개수 출력
print(f"Downsampled point count: {len(downsample_pcd.points)}")

# Statistical Outlier Removal (SOR) 적용
cl, ind = downsample_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
sor_pcd = downsample_pcd.select_by_index(ind)
print(f"Statistical Outlier Removal count: {len(sor_pcd.points)}")

# Radius Outlier Removal (ROR) 적용
cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.0)
ror_pcd = downsample_pcd.select_by_index(ind)
print(f"Radius Outlier Removalt: {len(ror_pcd.points)}")

# 각 포인트 클라우드에 색상 지정
original_pcd.paint_uniform_color([0, 0, 0])
downsample_pcd.paint_uniform_color([0, 0, 1])  # 빨강
sor_pcd.paint_uniform_color([1, 0, 0])  # 파랑
ror_pcd.paint_uniform_color([0, 1, 0])  # 녹색

# 세 포인트 클라우드를 함께 시각화

# o3d.visualization.draw_geometries([downsample_pcd], window_name="downsample_pcd")
# o3d.visualization.draw_geometries([sor_pcd], window_name="sor_pcd")
# o3d.visualization.draw_geometries([ror_pcd], window_name="ror_pcd")

o3d.visualization.draw_geometries(
    [original_pcd, downsample_pcd], 
    window_name="Original (Black), downsample_pcd (Red) Point Cloud"
)

o3d.visualization.draw_geometries(
    [downsample_pcd, sor_pcd], 
    window_name="Downsampled (Red), SOR (Blue) Point Cloud"
)

o3d.visualization.draw_geometries(
    [sor_pcd, ror_pcd], 
    window_name="SOR (Blue), ROR (Green) Point Cloud"
)

o3d.visualization.draw_geometries(
    [original_pcd, ror_pcd], 
    window_name="Original (Black), Outlier eleminate (Blue) Point Cloud"
)