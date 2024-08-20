from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# 추적 기록을 저장하기 위한 기본 빈 리스트를 갖는 딕셔너리 생성
track_history = defaultdict(lambda: [])

# YOLO 모델 로드 (객체 탐지 전용)
model = YOLO("/home/ubuntu/Desktop/sensor_fusion/camera_only/yolov8n.pt")

# 비디오 파일 열기
cap = cv2.VideoCapture("/home/ubuntu/Desktop/sensor_fusion/sensor_fusion/test_video.mp4")

# 비디오 속성 가져오기: 너비, 높이, 프레임당 재생 시간(초당 프레임 수)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# 지정된 속성으로 출력 비디오를 저장하기 위해 비디오 작성기 초기화
out = cv2.VideoWriter("object-tracking_result.mp4", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

while True:
    # 비디오에서 프레임 읽기
    ret, im0 = cap.read()
    if not ret:
        print("비디오 프레임이 비어있거나 비디오 처리가 성공적으로 완료되었습니다.")
        break

    # 프레임에 그리기 위한 Annotator 객체 생성
    annotator = Annotator(im0, line_width=2)

    # 현재 프레임에서 객체 추적 수행
    results = model.track(im0, persist=True)

    # 결과에 추적 ID와 경계 상자가 있는지 확인
    if results[0].boxes.id is not None:
        # 경계 상자와 추적 ID 추출
        boxes = results[0].boxes.xyxy
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # 각 경계 상자에 해당하는 추적 ID와 색상으로 주석 달기
        for box, track_id in zip(boxes, track_ids):
            # 바운딩 박스의 중심점 계산
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            
            # track_history에 중심점 추가
            track_history[track_id].append((center_x, center_y))

            # # 이전 중심점과 현재 중심점 사이의 선을 그림으로써 트랙렛을 그림
            # for j in range(1, len(track_history[track_id])):
            #     if track_history[track_id][j - 1] is None or track_history[track_id][j] is None:
            #         continue
            #     # 이전 점과 현재 점 사이의 선 그리기
            #     cv2.line(im0, track_history[track_id][j - 1], track_history[track_id][j], colors(track_id, True), 2)

            # 바운딩 박스와 ID 주석 달기
            annotator.box_label(box=box, label=str(track_id), color=colors(track_id, True))

    # 주석이 달린 프레임을 출력 비디오에 쓰기
    out.write(im0)
    # 주석이 달린 프레임을 화면에 표시
    cv2.imshow("instance-segmentation-object-tracking", im0)

    # 'q' 키가 눌리면 루프를 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 비디오 작성기와 캡처 객체 해제, 그리고 모든 OpenCV 창 닫기
out.release()
cap.release()
cv2.destroyAllWindows()
