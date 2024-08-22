import os
from glob import glob
import cv2
import numpy as np
import matplotlib.cm as cm
from sklearn import linear_model
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

# YOLOv8 설정
model = YOLO('/home/ubuntu/Desktop/object_detection/camera_only/yolov8n.pt')

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
    
    if remove_outliers and image is not None:
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

def get_3d_bboxes(cluster_dict, labels, velo_points):
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
            continue

        # 3D Bounding Box의 포인트들을 정의합니다.
        box_points = np.array([
            [x_max, y_max, z_max, 1], [x_max, y_max, z_min, 1],
            [x_max, y_min, z_max, 1], [x_max, y_min, z_min, 1],
            [x_min, y_max, z_max, 1], [x_min, y_max, z_min, 1],
            [x_min, y_min, z_max, 1], [x_min, y_min, z_min, 1]
        ])

        # 이 박스 포인트들을 이미지 공간으로 변환합니다.
        camera_box_points = T_mat @ box_points.T
        camera_box_points[:2] /= camera_box_points[2, :]
        camera_box_point_list.append(camera_box_points.round().T.astype(int).tolist())

    return camera_box_point_list

def get_2d_bboxes(camera_box_points):
    ''' 3D BBOX의 포인트들을 기반으로 가장 잘 포함하는 2D BBOX를 생성 '''
    camera_2d_box_points = []

    for box_pts in camera_box_points:
        u_coords = [pt[0] for pt in box_pts]
        v_coords = [pt[1] for pt in box_pts]

        u_min, v_min = min(u_coords), min(v_coords)
        u_max, v_max = max(u_coords), max(v_coords)

        camera_2d_box_points.append([[u_min, v_min], [u_max, v_max]])

    return camera_2d_box_points

def draw_2d_boxes(image, bboxes, color=[0, 255, 0], thickness=3):
    ''' 2D BBOX를 이미지에 그리기 '''
    for bbox in bboxes:
        if isinstance(bbox[0], list):  # LiDAR 바운딩 박스: [[u_min, v_min], [u_max, v_max]] 형태
            top_left = [int(bbox[0][0]), int(bbox[0][1])]
            bottom_right = [int(bbox[1][0]), int(bbox[1][1])]
        else:  # YOLO 바운딩 박스: [x_min, y_min, x_max, y_max] 형태
            top_left = [int(bbox[0]), int(bbox[1])]
            bottom_right = [int(bbox[2]), int(bbox[3])]

        cv2.rectangle(image, top_left, bottom_right, color, thickness=thickness)

    return image

def calculate_iou(box1, box2):
    ''' 두 박스의 IoU 계산 '''
    x1, y1 = max(box1[0][0], box2[0][0]), max(box1[0][1], box2[0][1])
    x2, y2 = min(box1[1][0], box2[1][0]), min(box1[1][1], box2[1][1])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[1][0] - box1[0][0]) * (box1[1][1] - box1[0][1])
    box2_area = (box2[1][0] - box2[0][0]) * (box2[1][1] - box2[0][1])

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def hungarian_matching(image_bbox_list, lidar_bbox_list):
    ''' 헝가리안 매칭 알고리즘을 사용하여 IoU가 0.65 이상인 박스들을 매칭 '''
    cost_matrix = np.zeros((len(image_bbox_list), len(lidar_bbox_list)))

    # 비용 행렬 계산
    for i, img_bbox in enumerate(image_bbox_list):
        for j, lidar_bbox in enumerate(lidar_bbox_list):
            iou = calculate_iou(img_bbox, lidar_bbox)
            print("iou : ", iou)
            if iou >= 0.5:
                cost_matrix[i, j] = 1 - iou  # IoU가 클수록 매칭 비용이 작아지도록 설정
            else:
                cost_matrix[i, j] = 1.0  # 매칭 불가능한 경우 비용을 1.0로 설정

    # 헝가리안 알고리즘을 사용하여 최적 매칭 수행
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 1.0:  # 매칭된 박스 간의 IoU가 0.65 이상일 때
            iou = 1 - cost_matrix[r, c]
            if iou > 0.6:  # 추가 조건: IoU가 0.6보다 큰 경우에만 매칭 리스트에 포함
                matches.append((r, c))

    return matches


def bbox_match(image_bbox_list, lidar_bbox_list, matches):
    ''' 매칭된 bbox들을 하나의 새로운 bbox로 생성 '''
    match_bbox_list = []
    unmatched_image_bbox_list = []
    unmatched_lidar_bbox_list = []

    used_image_indices = set()
    used_lidar_indices = set()

    for img_idx, lidar_idx in matches:
        if img_idx not in used_image_indices and lidar_idx not in used_lidar_indices:
            img_bbox = image_bbox_list[img_idx]
            lidar_bbox = lidar_bbox_list[lidar_idx]

            img_center = [(img_bbox[0][0] + img_bbox[1][0]) // 2, (img_bbox[0][1] + img_bbox[1][1]) // 2]
            lidar_center = [(lidar_bbox[0][0] + lidar_bbox[1][0]) // 2, (lidar_bbox[0][1] + lidar_bbox[1][1]) // 2]

            final_center = [(img_center[0] + lidar_center[0]) // 2, (img_center[1] + lidar_center[1]) // 2]

            final_width = (img_bbox[1][0] - img_bbox[0][0] + lidar_bbox[1][0] - lidar_bbox[0][0]) // 2
            final_height = (img_bbox[1][1] - img_bbox[0][1] + lidar_bbox[1][1] - lidar_bbox[0][1]) // 2

            final_bbox = [
                [final_center[0] - final_width // 2, final_center[1] - final_height // 2],
                [final_center[0] + final_width // 2, final_center[1] + final_height // 2]
            ]

            match_bbox_list.append(final_bbox)

            # 사용된 인덱스 기록
            used_image_indices.add(img_idx)
            used_lidar_indices.add(lidar_idx)

    # 매칭되지 않은 박스들을 추가로 관리
    for i, bbox in enumerate(image_bbox_list):
        if i not in used_image_indices:
            unmatched_image_bbox_list.append(bbox)

    for i, bbox in enumerate(lidar_bbox_list):
        if i not in used_lidar_indices:
            unmatched_lidar_bbox_list.append(bbox)

    return match_bbox_list, unmatched_image_bbox_list, unmatched_lidar_bbox_list


def draw_final_bbox(image, match_bbox_list, image_bbox_list, lidar_bbox_list):
    ''' 최종 BBOX들을 이미지에 그리기 '''
    image = draw_2d_boxes(image, match_bbox_list, color=[0, 255, 0], thickness=5)
    image = draw_2d_boxes(image, image_bbox_list, color=[255, 0, 0], thickness=2)
    image = draw_2d_boxes(image, lidar_bbox_list, color=[0, 0, 255], thickness=2)

    return image

def count_bbox(image, total_boxes):
    ''' 화면 좌측 상단에 BBOX 개수 출력 '''
    text = f'BBOX count : {total_boxes}'
    font_scale = 0.5
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = 10
    text_y = 10 + text_size[1]

    cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)

    return image

def main_pipeline(left_image, lidar_bin, velo_points):
    # YOLO를 통해 2D 바운딩 박스 추출 (빨간색)
    yolo_detections = model(left_image)
    yolo_bboxes = yolo_detections[0].boxes.xyxy.cpu().numpy()

    dbscan, labels = get_clusters(velo_points)
    cam_clusters = image_clusters_from_velo(velo_points, labels, left_image)
    camera_box_points = get_3d_bboxes(cam_clusters, labels, velo_points)
    lidar_bboxes = get_2d_bboxes(camera_box_points)

    # image_bbox_list와 lidar_bbox_list 생성
    image_bbox_list = [[[int(bbox[0]), int(bbox[1])], [int(bbox[2]), int(bbox[3])]] for bbox in yolo_bboxes]
    
    # 이미지 저장: YOLO 및 LiDAR 박스 별도
    image_infer_bbox = draw_2d_boxes(left_image.copy(), image_bbox_list, color=[255, 0, 0], thickness=2)
    cv2.imwrite('image_infer.png', cv2.cvtColor(image_infer_bbox, cv2.COLOR_RGB2BGR))

    lidar_infer_bbox = draw_2d_boxes(left_image.copy(), lidar_bboxes, color=[0, 0, 255], thickness=2)
    cv2.imwrite('lidar_infer.png', cv2.cvtColor(lidar_infer_bbox, cv2.COLOR_RGB2BGR))


    # 헝가리안 매칭 알고리즘을 통한 bbox 매칭
    matches = hungarian_matching(image_bbox_list, lidar_bboxes)
    match_bbox_list, image_bbox_list, lidar_bbox_list = bbox_match(image_bbox_list, lidar_bboxes, matches)

    # 최종 BBOX 그리기
    sensor_fusion_bbox = draw_final_bbox(left_image, match_bbox_list, image_bbox_list, lidar_bbox_list)
    
    # BBOX 개수 카운트 및 표시
    total_boxes = len(match_bbox_list) + len(image_bbox_list) + len(lidar_bbox_list)
    sensor_fusion_bbox = count_bbox(sensor_fusion_bbox, total_boxes)
    
    # 최종 이미지 저장
    cv2.imwrite('sensor_fusion_infer.png', cv2.cvtColor(sensor_fusion_bbox, cv2.COLOR_RGB2BGR))
    
    return image_infer_bbox, lidar_infer_bbox, sensor_fusion_bbox


index = 50
left_image = cv2.cvtColor(cv2.imread(left_image_paths[index]), cv2.COLOR_BGR2RGB)
lidar_bin = bin_paths[index]
velo_points = bin2h_velo(lidar_bin, remove_plane=True)

camera_infer_image, lidar_infer_image, sensor_fusion_image = main_pipeline(left_image, lidar_bin, velo_points)

cv2.imshow('image_infer_bbox', cv2.cvtColor(camera_infer_image, cv2.COLOR_RGB2BGR))
cv2.imshow('lidar_infer_bbox', cv2.cvtColor(lidar_infer_image, cv2.COLOR_RGB2BGR))
cv2.imshow('sensor_fusion_bbox', cv2.cvtColor(sensor_fusion_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
