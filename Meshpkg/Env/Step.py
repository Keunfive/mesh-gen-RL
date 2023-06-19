import numpy as np
import math

import Meshpkg.params as p
from Meshpkg.Env.Reward import get_reward
from Meshpkg.Calculation.checkcross import check_cross
from Meshpkg.Calculation import angle
from Meshpkg.Env.State import layer_to_state

class step_class:
  
  def __init__(self):
      self.volume_mesh = [p.surf_mesh]
      self.num_neighbor = p.num_neighbor
      self.cur_layer = 0
      self.length = p.surf_length
      self.first_layer = p.first_layer
      self.growth_rate = p.growth_rate
      self.max_layer = p.num_layer


  def get_frontal_mesh(self):
    return self.volume_mesh[-1]

  def step_func(self, action, step, episode = None):

      def step_in_layer (i, action_index):
          state = layer_to_state(self.volume_mesh[-1])[i]
          theta_2 = angle.get_angle2(state[self.num_neighbor+1], state[self.num_neighbor])
          theta_1 = angle.get_angle2(state[self.num_neighbor-1], state[self.num_neighbor])

          if theta_2 < theta_1 and theta_1 < 360:
            theta_2 += 360
            
          # txt_file.write(f'{action_index}-{len(p.action_space)} ')
          action_step = p.action_space[action_index]


          delta_margin = (theta_2 - theta_1)/(4) #4 등분
          delta_theta = (delta_margin * 2)/(5+1) # 4등분한거 중앙 두개 사이에서 6등분한 것 중 하나의 각도.
          
          ## 변경 부분 ##
          x = state[self.num_neighbor][0] + self.first_layer * pow(self.growth_rate, self.cur_layer) * (
            0.25 + 0.25*action_step[0]) * math.cos(math.radians(theta_1 + delta_margin + delta_theta*action_step[1]))
          y = state[self.num_neighbor][1] + self.first_layer * pow(self.growth_rate, self.cur_layer) * (
            0.25 + 0.25*action_step[0]) * math.sin(math.radians(theta_1 + delta_margin + delta_theta*action_step[1]))
          ##          ##
          next_point = [x,y]
          return next_point

      next_layer = [ ]
      info = set()


      for i in range(self.length):
        next_layer.append(step_in_layer(i, action[i]))


      self.volume_mesh.append(np.array(next_layer))
      next_state = layer_to_state(next_layer)
      r = get_reward(self.volume_mesh)
      reward = 0.1 * r.get_skew() + 0.6 *r.get_length_ratio()+ 0.3 *r.get_jacobian()[0]
      

      txt_file = open("reward_record.txt", 'a')
      if episode != None:
        if step == 1:
          txt_file.write(f'\n\n ----------reward report: episode: {episode}---------- \n\n')
        txt_file.write(f'step: [{step}] \n\n')
        txt_file.write(f'Skew:         \n{r.get_skew()}\n')
        txt_file.write(f'Length ratio: \n{r.get_length_ratio()}\n')              
        txt_file.write(f'Jacobian:     \n{r.get_jacobian()[0]}\n')
        txt_file.write(f'reward sum: \n\n{reward}\n\n')
        txt_file.close()
      """꼬임 여부 판별하는 부분"""

      for i in range(self.length):

        if (any(r.get_jacobian()[1][i]) < 0) or (r.get_skew()[i] < 0):
          info.add(i)
      is_crossed, crossed_pt = check_cross(self.volume_mesh[-1])
      info = info.union(crossed_pt)
      self.cur_layer += 1

      if self.cur_layer == self.max_layer:
        dones = np.ones(self.length)
      elif len(info) != 0:
        dones = np.ones(self.length)
      else:
        dones = np.zeros(self.length) 

      steps = step * np.ones(self.length)
      
      return next_state, reward, dones, info, steps

  def reset(self):
      self.volume_mesh = [p.surf_mesh]
      self.cur_layer = 0
      return layer_to_state(self.volume_mesh[-1]) #맨처음 state 반환