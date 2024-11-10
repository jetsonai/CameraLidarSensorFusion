import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import linear_sum_assignment

# 그룹 A의 박스 정의
a_boxes = {
    'a_box_01': {'x': 1, 'y': 1, 'width': 3, 'height': 4},
    'a_box_02': {'x': 2, 'y': 2, 'width': 4, 'height': 5},
    'a_box_03': {'x': 3, 'y': 3, 'width': 5, 'height': 3},
    'a_box_04': {'x': 5, 'y': 5, 'width': 4, 'height': 3},
}

# 그룹 B의 박스 정의
b_boxes = {
    'b_box_01': {'x': 3, 'y': 1, 'width': 3, 'height': 4},
    'b_box_02': {'x': 2, 'y': 5, 'width': 4, 'height': 5},
    'b_box_03': {'x': 1, 'y': 3, 'width': 5, 'height': 3},
    'b_box_04': {'x': 6, 'y': 6, 'width': 4, 'height': 3},
}

# IoU 계산 함수 정의
def calculate_iou(box_a, box_b):
    # 두 박스 간의 x, y 축에서의 겹치는 부분 계산
    x_overlap = max(0, min(box_a['x'] + box_a['width'], box_b['x'] + box_b['width']) - max(box_a['x'], box_b['x']))
    y_overlap = max(0, min(box_a['y'] + box_a['height'], box_b['y'] + box_b['height']) - max(box_a['y'], box_b['y']))
    intersection_area = x_overlap * y_overlap
    
    # 각 박스의 면적 계산
    a_box_area = box_a['width'] * box_a['height']
    b_box_area = box_b['width'] * box_b['height']
    
    # 두 박스의 합집합 영역 계산
    union_area = a_box_area + b_box_area - intersection_area
    
    # IoU 계산 (만약 합집합이 0이면 IoU도 0)
    if union_area == 0:
        return 0
    
    iou = intersection_area / union_area
    return iou

# IoU 행렬 초기화 (그룹 A와 B의 박스 수에 맞게)
iou_matrix = np.zeros((len(a_boxes), len(b_boxes)))

# IoU 행렬을 채움
for i, (_, a_box) in enumerate(a_boxes.items()):
    for j, (_, b_box) in enumerate(b_boxes.items()):
        iou_matrix[i, j] = calculate_iou(a_box, b_box)

# 헝가리안 알고리즘은 비용을 최소화하므로, IoU를 최대화하기 위해 음수로 변환
cost_matrix = -iou_matrix

# 헝가리안 알고리즘을 사용하여 최적의 할당을 계산
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# 최적의 할당 결과 및 IoU 출력
print("IoU를 최대화하기 위한 A와 B 박스 간의 최적 할당:")
for i, j in zip(row_ind, col_ind):
    print(f"A 박스 {i+1} -> B 박스 {j+1} (IoU: {iou_matrix[i, j]:.4f})")

# 전체 최대 IoU 계산 및 출력
total_max_iou = iou_matrix[row_ind, col_ind].sum()
print(f"\n전체 최대 IoU 합: {total_max_iou:.4f}")

# 최종 매칭된 박스들 간의 IoU 시각화
fig, axes = plt.subplots(1, len(row_ind), figsize=(15, 5))

for idx, (i, j) in enumerate(zip(row_ind, col_ind)):
    ax = axes[idx]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # A 박스와 B 박스를 시각화
    a_box = list(a_boxes.values())[i]
    b_box = list(b_boxes.values())[j]
    
    rect_a = patches.Rectangle((a_box['x'], a_box['y']), a_box['width'], a_box['height'], linewidth=3, edgecolor='r', facecolor='none', label="A 박스")
    rect_b = patches.Rectangle((b_box['x'], b_box['y']), b_box['width'], b_box['height'], linewidth=3, edgecolor='b', facecolor='none', label="B 박스")
    
    ax.add_patch(rect_a)
    ax.add_patch(rect_b)
    
    # 교집합 영역을 초록색으로 표시
    x_overlap = max(0, min(a_box['x'] + a_box['width'], b_box['x'] + b_box['width']) - max(a_box['x'], b_box['x']))
    y_overlap = max(0, min(a_box['y'] + a_box['height'], b_box['y'] + b_box['height']) - max(a_box['y'], b_box['y']))
    if x_overlap > 0 and y_overlap > 0:
        intersection_rect = patches.Rectangle(
            (max(a_box['x'], b_box['x']), max(a_box['y'], b_box['y'])),
            x_overlap, y_overlap, linewidth=0, edgecolor='none', facecolor='green', alpha=0.5
        )
        ax.add_patch(intersection_rect)
    
    # 각 박스 쌍에 대한 IoU 값과 제목 설정
    ax.set_title(f"A_box {i+1} vs B_box {j+1}\nIoU: {iou_matrix[i, j]:.4f}", fontsize=12)
    
    # 축 제거
    ax.set_xticks([])
    ax.set_yticks([])

# 레이아웃 조정 및 그래프 표시
plt.tight_layout()
plt.show()

