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

# 시각화
o3d.visualization.draw_geometries([original_pcd], window_name="Original Point Cloud")
o3d.visualization.draw_geometries([downsample_pcd], window_name="Downsampled Point Cloud")

# 두 포인트 클라우드를 함께 시각화
original_pcd.paint_uniform_color([1, 0, 0])  # 원본은 빨간색으로 표시
downsample_pcd.paint_uniform_color([0, 1, 0])  # 다운샘플링된 클라우드는 녹색으로 표시

o3d.visualization.draw_geometries([original_pcd, downsample_pcd], window_name="Original and Downsampled Point Cloud")