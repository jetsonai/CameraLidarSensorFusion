import numpy as np
from scipy.optimize import linear_sum_assignment # linear_sum_assignment : scipy에서 구현해놓은 hungarian metohd 함수

# 비용 행렬(cost matrix) 정의
# 4명의 작업자가 있고 
# 4개의 작업이 있으며 
# 각 작업자-작업 조합의 비용이 아래와 같다 가정
cost_matrix = np.array([
    [4, 2, 8, 5],
    [2, 3, 7, 2],
    [5, 8, 6, 4],
    [3, 7, 6, 3],
])

# 헝가리안 알고리즘을 사용하여 최적의 할당을 계산
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# 결과를 출력합니다.
print("작업자 -> 작업 할당:")
for i, j in zip(row_ind, col_ind):
    print(f"작업자 {i+1} -> 작업 {j+1} (비용: {cost_matrix[i, j]})")

# 총 최소 비용을 출력합니다.
total_cost = cost_matrix[row_ind, col_ind].sum()
print(f"\n총 최소 비용: {total_cost}")
