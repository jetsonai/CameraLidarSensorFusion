import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 명확성을 위해 상자를 딕셔너리로 정의
a_box = {'x': 2, 'y': 2, 'width': 4, 'height': 4}
b_box = {'x': 4, 'y': 4, 'width': 4, 'height': 4}

# 교차 영역 계산
x_overlap = max(0, min(a_box['x'] + a_box['width'], b_box['x'] + b_box['width']) - max(a_box['x'], b_box['x']))
y_overlap = max(0, min(a_box['y'] + a_box['height'], b_box['y'] + b_box['height']) - max(a_box['y'], b_box['y']))
intersection_area = x_overlap * y_overlap

# 합집합 영역 계산
a_box_area = a_box['width'] * a_box['height']
b_box_area = b_box['width'] * b_box['height']
union_area = a_box_area + b_box_area - intersection_area

# IoU 계산
iou = intersection_area / union_area

# 상자 그리기
fig, ax = plt.subplots(1)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# 합집합 영역 (A 또는 B에 의해 덮인 모든 영역)을 회색 패치로 그리기
union_rects = [
    patches.Rectangle((a_box['x'], a_box['y']), a_box['width'], a_box['height'], linewidth=0, edgecolor='none', facecolor='gray', alpha=0.3),
    patches.Rectangle((b_box['x'], b_box['y']), b_box['width'], b_box['height'], linewidth=0, edgecolor='none', facecolor='gray', alpha=0.3)
]

# 교차 영역을 초록색 패치로 그리기
if (intersection_area > 0):
    intersection_rect = patches.Rectangle(
        (max(a_box['x'], b_box['x']), max(a_box['y'], b_box['y'])),
        x_overlap, y_overlap, linewidth=0, edgecolor='none', facecolor='green', alpha=0.5
    )
    ax.add_patch(intersection_rect)

# A와 B의 사각형 생성
rect_a = patches.Rectangle((a_box['x'], a_box['y']), a_box['width'], a_box['height'], linewidth=3, edgecolor='r', facecolor='none', label="A Box")
rect_b = patches.Rectangle((b_box['x'], b_box['y']), b_box['width'], b_box['height'], linewidth=3, edgecolor='b', facecolor='none', label="B Box")

# 합집합 사각형을 플롯에 추가
for rect in union_rects:
    ax.add_patch(rect)

# A와 B 사각형을 플롯에 추가
ax.add_patch(rect_a)
ax.add_patch(rect_b)

# 범례 추가
plt.legend()

# 플롯 표시
plt.show()

# IoU 출력
print("IoU : ", iou)
