import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 그룹 A의 상자들을 정의
a_boxes = {
    'a_box_01': {'x': 1, 'y': 1, 'width': 3, 'height': 4},
    'a_box_02': {'x': 2, 'y': 2, 'width': 4, 'height': 5},
    'a_box_03': {'x': 3, 'y': 3, 'width': 5, 'height': 3},
    'a_box_04': {'x': 5, 'y': 5, 'width': 4, 'height': 3},
}

# 그룹 B의 상자들을 정의
b_boxes = {
    'b_box_01': {'x': 3, 'y': 1, 'width': 3, 'height': 4},
    'b_box_02': {'x': 2, 'y': 5, 'width': 4, 'height': 5},
    'b_box_03': {'x': 1, 'y': 3, 'width': 5, 'height': 3},
    'b_box_04': {'x': 6, 'y': 6, 'width': 4, 'height': 3},
}

# IoU(교집합 비율) 계산 함수
def calculate_iou(box_a, box_b):
    x_overlap = max(0, min(box_a['x'] + box_a['width'], box_b['x'] + box_b['width']) - max(box_a['x'], box_b['x']))
    y_overlap = max(0, min(box_a['y'] + box_a['height'], box_b['y'] + box_b['height']) - max(box_a['y'], box_b['y']))
    intersection_area = x_overlap * y_overlap
    
    a_box_area = box_a['width'] * box_a['height']
    b_box_area = box_b['width'] * box_b['height']
    union_area = a_box_area + b_box_area - intersection_area
    
    if union_area == 0:
        return 0
    
    iou = intersection_area / union_area
    return iou

# IoU 결과를 저장할 딕셔너리 초기화
iou_results = {}

# 플롯 생성
fig, axes = plt.subplots(len(a_boxes), len(b_boxes), figsize=(15, 15))

for i, (a_name, a_box) in enumerate(a_boxes.items()):
    for j, (b_name, b_box) in enumerate(b_boxes.items()):
        iou = calculate_iou(a_box, b_box)
        iou_results[f"{a_name}_{b_name}"] = iou
        
        # IoU 결과 출력
        print(f"IoU({a_name}, {b_name}): {iou:.4f}")
        
        # 각 상자 쌍을 플롯에 그리기
        ax = axes[i, j]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # 합집합 영역(A 또는 B에 의해 덮인 모든 영역)을 회색 패치로 그리기
        union_rects = [
            patches.Rectangle((a_box['x'], a_box['y']), a_box['width'], a_box['height'], linewidth=0, edgecolor='none', facecolor='gray', alpha=0.3),
            patches.Rectangle((b_box['x'], b_box['y']), b_box['width'], b_box['height'], linewidth=0, edgecolor='none', facecolor='gray', alpha=0.3)
        ]
        
        # 교집합 영역을 초록색 패치로 그리기
        x_overlap = max(0, min(a_box['x'] + a_box['width'], b_box['x'] + b_box['width']) - max(a_box['x'], b_box['x']))
        y_overlap = max(0, min(a_box['y'] + a_box['height'], b_box['y'] + b_box['height']) - max(a_box['y'], b_box['y']))
        if x_overlap > 0 and y_overlap > 0:
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
        
        # 상자 이름과 IoU 값을 제목으로 설정
        ax.set_title(f"{a_name} vs {b_name}\nIoU: {iou:.4f}", fontsize=10)
        
        # 축 레이블 제거
        ax.set_xticks([])
        ax.set_yticks([])

# 레이아웃 조정
plt.tight_layout()
plt.show()

# IoU 결과를 파일에 저장
with open("iou_results.txt", "w") as f:
    for key, value in iou_results.items():
        f.write(f"{key}: {value:.4f}\n")
