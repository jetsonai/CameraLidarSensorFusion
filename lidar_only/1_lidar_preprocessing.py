import os
import numpy as np
import open3d as o3d

def bin_to_pcd(bin_file_path, pcd_file_path):
    # .bin 파일 읽기
    lidar_points = np.fromfile(bin_file_path, dtype=np.float32)
    
    # 포인트가 4개 요소 (x, y, z, intensity)로 구성된 포인트 수로 재구성
    lidar_points = lidar_points.reshape(-1, 4)
    
    # x, y, z 좌표만 사용 (intensity는 사용하지 않음)
    lidar_points = lidar_points[:, :3]
    
    # Open3D PointCloud 객체 생성
    pcd = o3d.geometry.PointCloud()
    
    # 포인트 데이터를 Open3D 포맷으로 변환
    pcd.points = o3d.utility.Vector3dVector(lidar_points)
    
    # .pcd 파일로 저장
    o3d.io.write_point_cloud(pcd_file_path, pcd)

if __name__ == "__main__":
    # 변환할 파일의 경로 설정
    src_folder = "/home/ubuntu/Desktop/sensor_fusion/test_data/lidar"  # .bin 파일들이 있는 디렉토리 경로
    dest_folder = "/home/ubuntu/Desktop/sensor_fusion/lidar_only/pcd/"  # 변환된 .pcd 파일을 저장할 디렉토리 경로

    # 목적 디렉토리가 없으면 생성
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # .bin 파일들 변환
    for bin_file in os.listdir(src_folder):
        if bin_file.endswith(".bin"):
            bin_file_path = os.path.join(src_folder, bin_file)
            pcd_file_path = os.path.join(dest_folder, os.path.splitext(bin_file)[0] + ".pcd")
            print(f"Converting {bin_file} to {pcd_file_path}")
            bin_to_pcd(bin_file_path, pcd_file_path)
