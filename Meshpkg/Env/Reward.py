import numpy as np
import math

import Meshpkg.params as p
from Meshpkg.Calculation import angle, area

class get_reward():
   
    def __init__(self, volume_mesh):
        self.length = p.surf_length
        self.Q_eq = 90
        self.volume_mesh = volume_mesh
        self.num_neighbor = p.num_neighbor

    def get_skew(self):   # reward- skew 계산
        first_layer = self.volume_mesh[-1]
        second_layer = self.volume_mesh[-2]

        theta = [ ]
        skewness_avg = [ ]
        
        for i in range(self.length):
            theta_1 = (angle.get_angle3(first_layer[i][:], second_layer[i][:], second_layer[(i + 1) % (self.length)][:]))
            theta_2 = (angle.get_angle3(first_layer[(i + 1) % (self.length)][:], first_layer[i][:], second_layer[i][:]))
            theta_3 = (angle.get_angle3(second_layer[(i + 1) % (self.length)][:], first_layer[(i + 1) % (self.length)][:], first_layer[i][:]))
            theta_4 = (angle.get_angle3(second_layer[i][:], second_layer[(i + 1) % (self.length)][:], first_layer[(i + 1) % (self.length)][:]))
            theta.append([theta_1, theta_2, theta_3, theta_4])
        # 1-reward version
        skewness = [1 - (np.max([(np.max(theta[i]) - self.Q_eq) / (180 - self.Q_eq), (self.Q_eq - np.min(theta[i])) / self.Q_eq])) for i
                    in range(self.length)]
        # skewness_avg = [ (skewness[i-1] + skewness[i])/2 for i in range(self.length) ]

        "skewness 음수 계산 피하려는 시도"
        for i in range(self.length):
            x = (skewness[i-1] + skewness[i])/2
            if x >=0:
                skewness_avg.append(x)
            else:
                skewness_avg.append(0)
        
        return np.array(skewness_avg)

    def get_jacobian(self):

        Jacobian = []
        Jacobian_det = []
        for i in range(0, self.length):
            J_1 = [ [self.volume_mesh[-1][(i+1)%(self.length)][0] - self.volume_mesh[-1][i][0],  self.volume_mesh[-2][i][0] - self.volume_mesh[-1][i][0]],
                    [self.volume_mesh[-1][(i+1)%(self.length)][1] - self.volume_mesh[-1][i][1],  self.volume_mesh[-2][i][1] - self.volume_mesh[-1][i][1]] ]
            J_2 = [ [self.volume_mesh[-2][i][0] - self.volume_mesh[-1][i][0],   self.volume_mesh[-1][i-1][0] - self.volume_mesh[-1][i][0]],
                    [self.volume_mesh[-2][i][1] - self.volume_mesh[-1][i][1],   self.volume_mesh[-1][i-1][1] - self.volume_mesh[-1][i][1]] ]
            
            J_det = [np.linalg.det(J_1), np.linalg.det(J_2)]
            if any(x <= 0 for x in J_det):
                J = 0
            else: # J_det 이 무조건 둘다 양수여야 계산하도록. 또 min/max값 무조건 1이하 나오도록
                J = min(J_det)/max(J_det)
            Jacobian_det.append(J_det)
            Jacobian.append(J)

        return [np.array(Jacobian), Jacobian_det]

    def get_AR(self):

        AR = []

        for i in range(0, self.length):
            w_d = [self.volume_mesh[-2][i][0] - self.volume_mesh[-2][i-1][0], self.volume_mesh[-2][i][1] - self.volume_mesh[-2][i-1][1]]
            w_u = [self.volume_mesh[-1][i][0] - self.volume_mesh[-1][i-1][0], self.volume_mesh[-1][i][1] - self.volume_mesh[-1][i-1][1]]
            h_r = [self.volume_mesh[-1][i-1][0] - self.volume_mesh[-2][i-1][0], self.volume_mesh[-1][i-1][1] - self.volume_mesh[-2][i-1][1]]
            h_l = [self.volume_mesh[-1][i][0] - self.volume_mesh[-2][i][0], self.volume_mesh[-1][i][1] - self.volume_mesh[-2][i][1]]

            w1 = math.sqrt(math.pow(w_d[0], 2) + math.pow(w_d[1], 2))
            w2 = math.sqrt(math.pow(w_u[0], 2) + math.pow(w_u[1], 2))
            h1 = math.sqrt(math.pow(h_r[0], 2) + math.pow(h_r[1], 2))
            h2 = math.sqrt(math.pow(h_l[0], 2) + math.pow(h_l[1], 2))

            w_h = [ (w1 + w2) / 2, (h1 + h2) / 2 ]
            AR.append(min(w_h)/max(w_h))
        AR_avg = [(AR[(i + 1) % (self.length)] + AR[i]) / 2 for i in range(self.length) ]

        return np.array(AR_avg)
      


    def get_area_ratio(self): #reward-area ratio 계산

        Area_ratio = [ ]

        for i in range(0, self.length):
            "왼쪽 넓이"
            Area_left = area.getArea([self.volume_mesh[-1][i], self.volume_mesh[-1][i - 1], 
                                      self.volume_mesh[-2][i - 1], self.volume_mesh[-2][i]])
            "기준 넓이"
            Area = area.getArea([self.volume_mesh[-1][(i + 1) % (self.length)], self.volume_mesh[-1][i], 
                                 self.volume_mesh[-2][i], self.volume_mesh[-2][(i + 1) % (self.length)]])
            "오른쪽 넓이"
            Area_right = area.getArea([self.volume_mesh[-1][(i + 2) % (self.length)], self.volume_mesh[-1][(i + 1) % (self.length)], 
                                       self.volume_mesh[-2][(i + 1) % (self.length)], self.volume_mesh[-2][(i + 2)%(self.length)]])

            Area_neighbor = [Area_left, Area_right]
            # Area_ratio.append( 1 / ( max(Area/(min(Area_neighbor)), (max(Area_neighbor))/Area )- 1 + 0.001)  )
            Area_ratio.append( 1 / (max(Area / (min(Area_neighbor)), (max(Area_neighbor)) / Area)))

        Area_ratio_avg = [ (Area_ratio[i - 1] + Area_ratio[i]) / 2 for i in range(self.length) ]
        
        return np.array(Area_ratio_avg)
   
    def get_length_ratio(self):
        """
        Action에 의해 만들어지는 두개의 수평방향 길이 각각을 좌/우 num_neighbor 만큼의 수평방향 길이 평균 비교
        주변/min(기준), max(기준)/주변 중 더 안좋은 값으로 반영 
        """

        Length_ratio = [ ]

        for i in range(0, self.length):
            length_list = [ ]
            length_ref = [ ]
            for j in range(0, 2*(self.num_neighbor)):
                length_list.append(np.sqrt(np.sum( (self.volume_mesh[-1][(i-self.num_neighbor+j+1) % self.length, :] 
                                                    - self.volume_mesh[-1][(i-self.num_neighbor+j)%self.length, :])**2 )))
        
            length_ref = [length_list[self.num_neighbor-1], length_list[self.num_neighbor]]
            del length_list[(self.num_neighbor-1):(self.num_neighbor+1)]

            length_neighbor = np.mean(length_list)

            Length_ratio.append( 1 / 
                                (max (length_neighbor / (min(length_ref)), (max(length_ref)) / length_neighbor)) )
        
        return np.array(Length_ratio)

    def get_height_ratio(self):
        """
        [Action에 의해 정의되는 점]과 [Surface geometry 중점(max-min해서 1/2한 점)]사이의 거리(height)를 계산하여
        이 거리(height)가 같은 layer 내 거리들의 평균과 얼마나 떨어져있는지 
        0에서 1 사이 범위에서 비교: 거리(height)가 평균보다 작으면 height/평균, 더 크면 평균/height
        """
        X_coord = [self.volume_mesh[0][i][0] for i in range(self.length)]
        Y_coord = [self.volume_mesh[0][i][1] for i in range(self.length)]
        
        X_center = min(X_coord) + (max(X_coord) - min(X_coord))/2
        Y_center = min(Y_coord) + (max(Y_coord) - min(Y_coord))/2

        height_list = [ ]
        
        for j in range(self.length):
            height_list.append(np.sqrt(np.sum( (self.volume_mesh[-1][j, :] 
                                                    - [X_center, Y_center])**2 )))

        height_ref = np.mean(height_list)

        height_ratio = [1 - np.abs(height_list[k] - height_ref) / height_ref for k in range(self.length)]

        return np.array(height_ratio)