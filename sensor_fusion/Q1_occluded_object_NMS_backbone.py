import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.optimize import linear_sum_assignment

# 박스 정의
a_box = {'x': 20, 'y': 20, 'width': 40, 'height': 40}
b_box = {'x': 20, 'y': 15, 'width': 35, 'height': 45}
c_box = {'x': 15, 'y': 20, 'width': 45, 'height': 35}
main_boxes = [a_box, b_box, c_box]

# Anchor 박스 생성 (주변에 위치하도록)
def generate_anchor_boxes():
    anchor_boxes = [
        {'x': 16, 'y': 16, 'width': 35, 'height': 44},
        {'x': 18, 'y': 20, 'width': 34, 'height': 43},
        {'x': 22, 'y': 18, 'width': 45, 'height': 38},
        {'x': 20, 'y': 24, 'width': 42, 'height': 36},
        {'x': 17, 'y': 19, 'width': 39, 'height': 44},
        {'x': 23, 'y': 21, 'width': 34, 'height': 35},
        {'x': 19, 'y': 22, 'width': 40, 'height': 39},
        {'x': 21, 'y': 17, 'width': 43, 'height': 48},
        {'x': 15, 'y': 23, 'width': 46, 'height': 37},
        {'x': 24, 'y': 16, 'width': 38, 'height': 42},
    ]
    return anchor_boxes

anchor_boxes = generate_anchor_boxes()

######### IoU 리스트 생성 함수 #########################################################################################
def calculate_all_ious(main_boxes, anchor_boxes):
    def calculate_iou(box1, box2):
        """
        이부분 코드 작성
        """
        return iou

    # 각 메인 박스에 대해 앵커 박스들과의 IoU 계산
    iou_list = [[calculate_iou(main_box, anchor_box) for anchor_box in anchor_boxes] for main_box in main_boxes]
    return iou_list

# a_box, b_box, c_box와 anchor_box들의 IoU 리스트 계산
iou_list = calculate_all_ious(main_boxes, anchor_boxes)

#####################################################################################################################




### 헝가리안 알고리즘을 사용하여 최적의 매칭을 찾는 함수########################################################################

def find_best_matching(iou_list):

    """
    이부분 코드 작성
    """
    
    return col_indices # 매칭 쌍 생성

#####################################################################################################################



#매칭 결과를 시각화하는 함수
def visualize_matching(main_boxes, anchor_boxes, col_indices):
    fig, ax = plt.subplots()

    colors = ['red', 'blue', 'green']  # 각 박스에 대응되는 색상
    for i, main_box in enumerate(main_boxes):
        ax.add_patch(patches.Rectangle((main_box['x'], main_box['y']), main_box['width'], main_box['height'], 
                                       linewidth=3, edgecolor=colors[i], facecolor=colors[i], alpha=0.3, label=f'box_{i+1}'))
        # 매칭된 앵커 박스 그리기
        anchor_box = anchor_boxes[col_indices[i]]
        ax.add_patch(patches.Rectangle((anchor_box['x'], anchor_box['y']), anchor_box['width'], anchor_box['height'], 
                                       linewidth=2, edgecolor=colors[i], facecolor='none', linestyle='--'))

    # 축 설정 및 그래프 표시
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 70)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()  # 이미지 좌표 시스템과 맞추기 위해 Y축을 반전
    plt.show()

# 최적의 매칭 찾기
col_indices = find_best_matching(iou_list) # 정답이 이렇게 나와야 합니다 : col_indices [6 0 8]

# 매칭 결과 시각화
visualize_matching(main_boxes, anchor_boxes, col_indices)
