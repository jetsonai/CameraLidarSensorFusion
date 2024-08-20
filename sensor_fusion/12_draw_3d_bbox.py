import os
from glob import glob
import cv2
import numpy as np
from sklearn import linear_model
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm
from ultralytics import YOLO

# YOLOv8 설정
model = YOLO('/home/ubuntu/Desktop/sensor_fusion/camera_only/yolov8n.pt')

DATA_PATH = r'2011_09_26/2011_09_26_drive_0106_sync'
left_image_paths = sorted(glob(os.path.join(DATA_PATH, 'image_02/data/*.png')))
bin_paths = sorted(glob(os.path.join(DATA_PATH, 'velodyne_points/data/*.bin')))

# 카메라 보정 데이터 가져오기
with open('2011_09_26/calib_cam_to_cam.txt', 'r') as f:
    calib = f.readlines()

P_left = np.array([float(x) for x in calib[25].strip().split(' ')[1:]]).reshape((3, 4))
R_left_rect = np.array([float(x) for x in calib[24].strip().split(' ')[1:]]).reshape((3, 3))
R_left_rect = np.insert(R_left_rect, 3, values=[0, 0, 0], axis=0)
R_left_rect = np.insert(R_left_rect, 3, values=[0, 0, 0, 1], axis=1)

# LiDAR에서 카메라로의 회전 및 변환 행렬 가져오기
with open(r'2011_09_26/calib_velo_to_cam.txt', 'r') as f:
    calib = f.readlines()
R_cam_velo = np.array([float(x) for x in calib[1].strip().split(' ')[1:]]).reshape((3, 3))
t_cam_velo = np.array([float(x) for x in calib[2].strip().split(' ')[1:]])[:, None]

T_cam_velo = np.vstack((np.hstack((R_cam_velo, t_cam_velo)), np.array([0, 0, 0, 1])))

# LiDAR 포인트를 카메라 좌표로 변환하기 위한 행렬 계산
T_mat = P_left @ R_left_rect @ T_cam_velo

def velo2camera(velo_points, image=None, remove_outliers=True):
    ''' LiDAR 포인트를 카메라(u,v,z) 공간으로 맵핑 '''
    velo_camera = T_mat @ velo_points
    velo_camera = np.delete(velo_camera, np.where(velo_camera[2, :] < 0)[0], axis=1)
    velo_camera[:2] /= velo_camera[2, :]
    
    if remove_outliers:
        u, v, z = velo_camera
        img_h, img_w, _ = image.shape
        u_out = np.logical_or(u < 0, u > img_w)
        v_out = np.logical_or(v < 0, v > img_h)
        outlier = np.logical_or(u_out, v_out)
        velo_camera = np.delete(velo_camera, np.where(outlier), axis=1)

    return velo_camera

def bin2h_velo(lidar_bin, remove_plane=True):
    ''' LiDAR bin 파일을 읽어 균질 좌표 (x,y,z,1)의 LiDAR 포인트를 반환 '''
    scan_data = np.fromfile(lidar_bin, dtype=np.float32).reshape((-1, 4))
    velo_points = scan_data[:, 0:3]

    if remove_plane:
        ransac = linear_model.RANSACRegressor(
            linear_model.LinearRegression(),
            residual_threshold=0.1,
            max_trials=5000
        )
        X = velo_points[:, :2]
        y = velo_points[:, -1]
        ransac.fit(X, y)
        velo_points = velo_points[~ransac.inlier_mask_]

    return np.insert(velo_points, 3, 1, axis=1).T

def project_velo2cam(lidar_bin, image, remove_plane=True):
    ''' LiDAR 포인트 클라우드를 이미지 좌표 프레임에 투영 '''
    velo_points = bin2h_velo(lidar_bin, remove_plane)
    return velo2camera(velo_points, image, remove_outliers=True)

def draw_clusters_on_image(image, cluster_dict):
    ''' 클러스터를 이미지 위에 그리기 '''
    pastel = cm.get_cmap('Pastel2', lut=50)
    get_pastel = lambda z : [255*val for val in pastel(int(z.round()))[:3]]

    for label, cluster in cluster_dict.items():
        for (u, v, z) in cluster.T:
            cv2.circle(image, (int(u), int(v)), 1, get_pastel(label), -1)

    return image

def get_clusters(velo_points):
    ''' LiDAR 클러스터링 '''
    dbscan = DBSCAN(eps=0.5, min_samples=30)
    dbscan.fit(velo_points[:3, :].T)
    return dbscan, dbscan.labels_

def image_clusters_from_velo(velo_points, labels, image):
    ''' 클러스터를 이미지로 변환 '''
    cam_clusters = {}
    for label in np.unique(labels):
        velo_cam = velo2camera(velo_points[:, labels == label], image)
        if velo_cam.shape[1] > 0:
            cam_clusters[label] = velo_cam
    return cam_clusters

def get_depth_detections(left_image, lidar_bin):
    ''' 객체 감지 및 깊이 계산 '''
    detections = model(left_image)
    bboxes = detections[0].boxes.xyxy.cpu().numpy()
    velo_camera = project_velo2cam(lidar_bin, left_image)
    return bboxes, velo_camera

import numpy as np

def get_3d_bboxes(cluster_dict, labels, velo_points, z_threshold=10):
    ''' 3D BBOX 생성 및 각 박스의 크기 출력 '''
    camera_box_point_list = []

    for c_label, cluster in cluster_dict.items():
        velo_cluster = velo_points[:3, labels == c_label]

        # 노이즈 클러스터 필터링
        if velo_cluster.shape[1] < 40:  # 최소 포인트 개수 필터링
            continue

        x_min, y_min, z_min = velo_cluster.min(axis=1)
        x_max, y_max, z_max = velo_cluster.max(axis=1)

        # Bounding Box 크기 계산
        width = x_max - x_min
        depth = y_max - y_min
        height = z_max - z_min

        # 크기 필터링: width, depth, height 중 하나라도 일정값 기준 이상 이하 경우 제외
        if width >= 5 or width <= 0.5  or depth >= 3  or depth <= 0.5  or height >= 1.5 or height <= 0.75:
            # print(f"Cluster {c_label} 제외됨: Width = {width:.2f}, depth = {depth:.2f}, height = {height:.2f}")
            continue

        print(f"Cluster {c_label}: Width = {width:.2f}, depth = {depth:.2f}, height = {height:.2f}, z_max = {z_max:.2f}")

        # 이제 3D Bounding Box의 포인트들을 정의합니다.
        box_points = np.array([
            [x_max, y_max, z_max, 1], [x_max, y_max, z_min, 1],
            [x_max, y_min, z_max, 1], [x_max, y_min, z_min, 1],
            [x_min, y_max, z_max, 1], [x_min, y_max, z_min, 1],
            [x_min, y_min, z_max, 1], [x_min, y_min, z_min, 1]
        ])

        # 이 박스 포인트들을 이미지 공간으로 변환합니다.
        camera_box_points = T_mat @ box_points.T
        camera_box_points[:2] /= camera_box_points[2, :]
        camera_box_point_list.append(camera_box_points.round().T.astype(int))

    return camera_box_point_list

def draw_3d_boxes(image, camera_box_points):
    ''' 3D 박스를 이미지에 그리기 '''
    pastel = cm.get_cmap('Pastel2', lut=50)
    get_pastel = lambda z : [255*val for val in pastel(z)[:3]]

    for i, box_pts in enumerate(camera_box_points):
        [A, B, C, D, E, F, G, H] = box_pts[:, :2]
        color = get_pastel(i)

        cv2.line(image, A, B, color, 2)
        cv2.line(image, B, D, color, 2)
        cv2.line(image, A, C, color, 2)
        cv2.line(image, D, C, color, 2)
        cv2.line(image, G, E, color, 2)
        cv2.line(image, H, F, color, 2)
        cv2.line(image, G, H, color, 2)
        cv2.line(image, E, F, color, 2)
        cv2.line(image, E, A, color, 2)
        cv2.line(image, G, C, color, 2)
        cv2.line(image, F, B, color, 2)
        cv2.line(image, H, D, color, 2)

    return image

def main_pipeline(left_image, lidar_bin, velo_points):
    bboxes, velo_camera = get_depth_detections(left_image, lidar_bin)
    dbscan, labels = get_clusters(velo_points)
    cam_clusters = image_clusters_from_velo(velo_points, labels, left_image)
    camera_box_points = get_3d_bboxes(cam_clusters, labels, velo_points)
    left_image = draw_3d_boxes(left_image, camera_box_points)
    return left_image, cam_clusters

index = 50
left_image = cv2.cvtColor(cv2.imread(left_image_paths[index]), cv2.COLOR_BGR2RGB)
lidar_bin = bin_paths[index]
velo_points = bin2h_velo(lidar_bin, remove_plane=True)

bbox_image, cam_clusters = main_pipeline(left_image, lidar_bin, velo_points)

cv2.imshow('3D BBOX Projected Points', cv2.cvtColor(bbox_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
