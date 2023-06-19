import numpy as np
import math

import Meshpkg.params as p

def layer_to_state(layer):
    state = []

    for i in range(len(layer)):
      state.append([layer[(i - p.num_neighbor + j) % len(layer)] for j in range(p.num_neighbor * 2 + 1)])

    return state

"get_new_state_2: 점 2개 없애는 함수"
def get_new_state_2(x_state):
    x_state = np.array(x_state)
    new_state = np.zeros(x_state.shape)

    for i in range(len(x_state)):
        new_state[i, :, :] = x_state[i, :, :] - x_state[i, p.num_neighbor, :]  # 기준점을 [0,0]으로 정렬

        theta_rot = math.atan2(new_state[i, p.num_neighbor + 1, 1],
                                new_state[i, p.num_neighbor + 1, 0])  # 회전할 각도 계산 -pi ~ pi 범위
        if theta_rot < 0:
            theta_rot += 2 * math.pi

        for j in range(p.num_neighbor * 2 + 1):
            if j != p.num_neighbor:
                x_ref = new_state[i, j, 0]
                y_ref = new_state[i, j, 1]
                #회전 변환 실행
                new_state[i, j, 0] = math.cos(theta_rot) * x_ref + math.sin(theta_rot) * y_ref
                new_state[i, j, 1] = - math.sin(theta_rot) * x_ref + math.cos(theta_rot) * y_ref

        length_ratio = 1 / np.sqrt(pow(new_state[i, p.num_neighbor + 1, 0], 2) + pow(new_state[i, p.num_neighbor + 1, 1], 2))

        for k in range(p.num_neighbor * 2 + 1):
            if k != p.num_neighbor:
                new_state[i, k, 0] = length_ratio * new_state[i, k, 0]
                new_state[i, k, 1] = length_ratio * new_state[i, k, 1]

    new_state = np.round(new_state, 4)

    # (0,0), (1,0)삭제
    new_state = np.delete(new_state, p.num_neighbor, axis=1)
    new_state = np.delete(new_state, p.num_neighbor, axis=1)

    new_state = new_state.reshape(len(new_state), -1)

    return new_state

"get_new_state_2: 점 2개 없애고 current time step 정보 집어넣는 함수"
def get_new_state_3(x_state, step):
    x_state = np.array(x_state)
    new_state = np.zeros(x_state.shape)

    for i in range(len(x_state)):
        new_state[i, :, :] = x_state[i, :, :] - x_state[i, p.num_neighbor, :]  # 기준점을 [0,0]으로 정렬

        theta_rot = math.atan2(new_state[i, p.num_neighbor + 1, 1],
                                new_state[i, p.num_neighbor + 1, 0])  # 회전할 각도 계산 -pi ~ pi 범위
        if theta_rot < 0:
            theta_rot += 2 * math.pi

        for j in range(p.num_neighbor * 2 + 1):
            if j != p.num_neighbor:
                x_ref = new_state[i, j, 0]
                y_ref = new_state[i, j, 1]
                #회전 변환 실행
                new_state[i, j, 0] = math.cos(theta_rot) * x_ref + math.sin(theta_rot) * y_ref
                new_state[i, j, 1] = - math.sin(theta_rot) * x_ref + math.cos(theta_rot) * y_ref

        length_ratio = 1 / np.sqrt(pow(new_state[i, p.num_neighbor + 1, 0], 2) + pow(new_state[i, p.num_neighbor + 1, 1], 2))

        for k in range(p.num_neighbor * 2 + 1):
            if k != p.num_neighbor:
                new_state[i, k, 0] = length_ratio * new_state[i, k, 0]
                new_state[i, k, 1] = length_ratio * new_state[i, k, 1]

    new_state = np.round(new_state, 4)

    # (0,0), (1,0)삭제
    new_state = np.delete(new_state, p.num_neighbor, axis=1)
    new_state = np.delete(new_state, p.num_neighbor, axis=1)

    new_state = np.append(new_state.reshape(len(new_state), -1) , 0.1 * step.reshape(len(new_state), -1), axis=1)

    return new_state

"get_new_state_1: 점 1개 없애기만 하는 함수"
def get_new_state_1(x_state, cur_layer):

    x_state = np.array(x_state)
    new_state = np.zeros(x_state.shape)

    for i in range(len(x_state)):
        new_state[i, :, :] = x_state[i, :, :] - x_state[i,p.num_neighbor, :]  # 기준점을 [0,0]으로 정렬

    new_state = np.round(new_state, 4)

    # (0,0) 삭제
    new_state = np.delete(new_state, p.num_neighbor, axis=1)
    new_state = np.append(new_state.reshape(len(x_state), -1) , cur_layer*np.ones(shape=(len(x_state), 1)), axis=1)

    return new_state

"get_new_state_1: 점 1개 없애고 [0,1]사이 normalize"
def get_new_state_1s(x_state):
    x_state = np.array(x_state)
    new_state = np.zeros(x_state.shape)

    for i in range(len(x_state)):
        new_state[i, :, :] = x_state[i, :, :] - x_state[i, p.num_neighbor, :]  # 기준점을 [0, 0]로 정렬

        maxmin_ratio = 1 / max( max(new_state[i, :, 0]) - min(new_state[i, :, 0]), max(new_state[i, :, 1]) - min(new_state[i, :, 1]) )  

        for k in range(p.num_neighbor * 2 + 1):
            if k != p.num_neighbor:
                new_state[i, k, 0] = maxmin_ratio * new_state[i, k, 0]
                new_state[i, k, 1] = maxmin_ratio * new_state[i, k, 1]

    new_state = np.round(new_state, 4)

    # (0,0) 삭제
    new_state = np.delete(new_state, p.num_neighbor, axis=1)
    new_state = new_state.reshape(len(new_state), -1)

    return new_state