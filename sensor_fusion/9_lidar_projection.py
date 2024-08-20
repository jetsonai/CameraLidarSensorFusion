import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

DATA_PATH = r'2011_09_26/2011_09_26_drive_0106_sync'

left_image_paths = sorted(glob(os.path.join(DATA_PATH, 'image_02/data/*.png')))
right_image_paths = sorted(glob(os.path.join(DATA_PATH, 'image_03/data/*.png')))

# LiDAR 데이터 가져오기
bin_paths = sorted(glob(os.path.join(DATA_PATH, 'velodyne_points/data/*.bin')))

print(f"Number of LiDAR Point: {len(bin_paths)}")

######################################################################################################
# 카메라 보정 데이터 가져오기
with open('2011_09_26/calib_cam_to_cam.txt','r') as f:
    calib = f.readlines()

# 투영 행렬 가져오기
P_left = np.array([float(x) for x in calib[25].strip().split(' ')[1:]]).reshape((3,4))
P_right = np.array([float(x) for x in calib[33].strip().split(' ')[1:]]).reshape((3,4))

# 교정된 회전 행렬 가져오기
# 두 개의 카메라를 통한 이미지 교정
R_left_rect = np.array([float(x) for x in calib[24].strip().split(' ')[1:]]).reshape((3, 3,))
R_right_rect = np.array([float(x) for x in calib[32].strip().split(' ')[1:]]).reshape((3, 3,))

R_left_rect = np.insert(R_left_rect, 3, values=[0,0,0], axis=0)
R_left_rect = np.insert(R_left_rect, 3, values=[0,0,0,1], axis=1)

def decompose_projection_matrix(P):    
    K, R, T, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    T = T/T[3]

    return K, R, T

K_left, R_left, T_left = decompose_projection_matrix(P_left)
K_right, R_right, T_right = decompose_projection_matrix(P_right)

######################################################################################################
# LiDAR에서 카메라로의 회전 및 변환 행렬 가져오기

with open(r'2011_09_26/calib_velo_to_cam.txt', 'r') as f:
    calib = f.readlines()

R_cam_velo = np.array([float(x) for x in calib[1].strip().split(' ')[1:]]).reshape((3, 3))
t_cam_velo = np.array([float(x) for x in calib[2].strip().split(' ')[1:]])[:, None]

T_cam_velo = np.vstack((np.hstack((R_cam_velo, t_cam_velo)),
                        np.array([0, 0, 0, 1])))

# 왼쪽 이미지를 사용할 것이므로, LiDAR 포인트를 왼쪽 이미지로 회전시키는 함수를 만듭니다.
T_mat = P_left @ R_left_rect @ T_cam_velo
T_mat

######################################################################################################
# LiDAR 포인트 파이프라인 생성

def velo2camera(velo_points, image=None, remove_outliers=True):
    ''' LiDAR 포인트를 카메라(u,v,z) 공간으로 맵핑합니다 '''
    # (왼쪽) 카메라 좌표로 변환
    # P_left @ R_left_rect @ T_cam_velo
    velo_camera =  T_mat @ velo_points

    # 음수 카메라 포인트 삭제
    velo_camera  = np.delete(velo_camera , np.where(velo_camera [2,:] < 0)[0], axis=1) 

    # 카메라 좌표 u,v,z 얻기
    velo_camera[:2] /= velo_camera[2, :]

    # 이상점 제거 (이미지 프레임 밖의 포인트)
    if remove_outliers:
        u, v, z = velo_camera
        img_h, img_w, _ = image.shape
        u_out = np.logical_or(u < 0, u > img_w)
        v_out = np.logical_or(v < 0, v > img_h)
        outlier = np.logical_or(u_out, v_out)
        velo_camera = np.delete(velo_camera, np.where(outlier), axis=1)

    return velo_camera

def bin2h_velo(lidar_bin, remove_plane=True):
    ''' LiDAR bin 파일을 읽어 균질 좌표 (x,y,z,1)의 LiDAR 포인트를 반환합니다 '''
    # LiDAR 데이터 읽기
    scan_data = np.fromfile(lidar_bin, dtype=np.float32).reshape((-1,4))

    # x,y,z LiDAR 포인트 얻기 (x, y, z) --> (앞, 왼쪽, 위)
    velo_points = scan_data[:, 0:3] 

    # 음수 LiDAR 포인트 삭제
    velo_points = np.delete(velo_points, np.where(velo_points[3, :] < 0), axis=1)

    # RANSAC을 사용해 평면 제거
    if remove_plane:
            ransac = linear_model.RANSACRegressor(
                                          linear_model.LinearRegression(),
                                          residual_threshold=0.1,
                                          max_trials=5000
                                          )

            X = velo_points[:, :2]
            y = velo_points[:, -1]
            ransac.fit(X, y)

            # 이상점 제거
            mask = ransac.inlier_mask_
            velo_points = velo_points[~mask]

    # 균질 LiDAR 포인트
    velo_points = np.insert(velo_points, 3, 1, axis=1).T 

    return velo_points

def project_velo2cam(lidar_bin, image, remove_plane=True):
    ''' LiDAR 포인트 클라우드를 이미지 좌표 프레임에 투영합니다 '''

    # bin 파일에서 균질 LiDAR 포인트 가져오기
    velo_points = bin2h_velo(lidar_bin, remove_plane)

    # 카메라 (u, v, z) 좌표 얻기
    velo_camera = velo2camera(velo_points, image, remove_outliers=True)
    
    return velo_camera

# LiDAR 투영 이미지 생성

index = 120

left_image = cv2.cvtColor(cv2.imread(left_image_paths[index]), cv2.COLOR_BGR2RGB)
lidar_bin = bin_paths[index]

# (u, v, z) = project_velo2cam(lidar_bin, left_image)
(u, v, z) = project_velo2cam(lidar_bin, left_image, remove_plane=False)

# z값 정규화 (0~255 범위)
z_normalized = (z - min(z)) / (max(z) - min(z)) * 255
z_normalized = z_normalized.astype(np.uint8)

# 포인트를 이미지에 표시
for i in range(len(u)):
    # z 값을 기반으로 색상 설정
    color = cv2.applyColorMap(np.array([[z_normalized[i]]], dtype=np.uint8), cv2.COLORMAP_RAINBOW)[0][0]
    cv2.circle(left_image, (int(u[i]), int(v[i])), 1, color.tolist(), -1)

# 이미지를 OpenCV의 imshow를 사용해 보여주기
cv2.imshow('Projected Points', cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR))  # RGB에서 BGR로 변환 필요
cv2.waitKey(0)
cv2.destroyAllWindows()
