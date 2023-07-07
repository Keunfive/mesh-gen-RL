import numpy as np

def check_cross_1(arr: np.array):
       """ 선분이 꼬이는 위치 출력 및 꼬임 여부 반환
           return 1:꼬임 있음, 0:꼬임 없음

           ex) check_cross()
           #>>> (1,2) 1번이랑 2번 선분, (32,33) 32번이랑 33번 선분 # 꼬이는 좌표 출력 
           #>>> 1 # 꼬임 여부

           *0411수정: crossed_pt_list.append -> crossed_pt_list.extend
       """
       crossed_pt_list=[]
       np_array = new_pt(arr)
       flag = 0
       for i in range(len(np_array)):
              for j in range(i, len(np_array)):
                     if is_cross_pt(*np_array[i], *np_array[j]) == 1:
                            # print(f"({i},{i + 1}), ({j},{j + 1})")
                            crossed_pt_list.extend([i,i+1,j,j+1])
                            
                            flag = 1
       if flag:
              return 1, crossed_pt_list
       
       return 0, crossed_pt_list


def check_cross_2(first_layer, second_layer): # 한 Layer에서 Action 실행시 (b_1 -> b_2) 이전 Action (a_1 -> a_2)과 꼬이지 않는지 여부를 확인하는 함수

       def ccw(p1, p2, p3):
              return (p2[0] - p1[0]) * (p3[1] - p1[1]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

       def intersect(a1, a2, b1, b2):
              # check if the segments (a1, a2) and (b1, b2) intersect
              ccw_a1 = ccw(a1, b1, b2)
              ccw_a2 = ccw(a2, b1, b2)
              ccw_b1 = ccw(b1, a1, a2)
              ccw_b2 = ccw(b2, a1, a2)
              return (ccw_a1 * ccw_a2 <= 0) and (ccw_b1 * ccw_b2 <= 0)
       crossed_pt_list_2 = [ ]
       for i in range(len(first_layer)):
              if intersect(second_layer[i - 1], first_layer[i - 1], second_layer[i], first_layer[i]):
                     crossed_pt_list_2.extend([i-1, i])
       if len(crossed_pt_list_2) == 0:
              return 0, crossed_pt_list_2
       else:
              return 1, crossed_pt_list_2


def new_pt(arr: np.array) -> np.array:
       """ 연속된 두 점을 선분의 형태로 바꾸는 함수 """
       return np.concatenate([arr[:-1], arr[1:]], axis=1)


""" 아래는 선분의 교차 여부를 확인하는 함수 """
def is_divide_pt(x11, y11, x12, y12, x21, y21, x22, y22) -> bool:
       '''
       input: 4 points
       output: True/False
       '''
       #  // line1 extension이 line2의 두 점을 양분하는지 검사..
       # 직선의 양분 판단
       f1 = (x12 - x11) * (y21 - y11) - (y12 - y11) * (x21 - x11)
       f2 = (x12 - x11) * (y22 - y11) - (y12 - y11) * (x22 - x11)
       if f1 * f2 < 0:
              return True
       else:
              return False

def is_cross_pt(x11, y11, x12, y12, x21, y21, x22, y22) -> int:
       b1 = is_divide_pt(x11, y11, x12, y12, x21, y21, x22, y22)
       b2 = is_divide_pt(x21, y21, x22, y22, x11, y11, x12, y12)
       if b1 and b2:
              return 1
       return 0


if __name__ == "__main__":
       """ test case """
       points = np.array([[1.44072409, -0.0051934],
                          [1.47657403, -0.02083323],
                          [1.50939523, 0.10464312],
                          [1.56337634, 0.15671139],
                          [1.61858929, 0.25228439],
                          [1.68692991, 0.35412849],
                          [1.78472497, 0.47851012],
                          [1.82468528, 0.70687353],
                          [1.73572394, 0.90076488],
                          [1.56161104, 1.08470643],
                          [1.38899121, 1.15604875],
                          [1.14974332, 1.17758051],
                          [0.83697969, 1.1500732],
                          [0.67856112, 1.08261268],
                          [0.3982351, 0.89390676],
                          [0.14151818, 0.72251852],
                          [-0.02828366, 0.41151395],
                          [-0.192093, 0.02445793],
                          [-0.0798573, -0.41001793],
                          [0.20150856, -0.68304711],
                          [0.39233185, -0.91151942],
                          [0.65509038, -1.02993329],
                          [0.90853435, -1.15483573],
                          [1.06546696, -1.17606294],
                          [1.33874756, -1.1623583],
                          [1.61503745, -1.05807838],
                          [1.71315298, -0.93771023],
                          [1.84210032, -0.74525478],
                          [1.78432724, -0.46739639],
                          [1.68223271, -0.35964004],
                          [1.59154018, -0.27091631],
                          [1.57466943, -0.17744762],
                          [1.52426146, -0.10041831],
                          [1.47601587, 0.01985301]])

       check = check_cross_1(points)