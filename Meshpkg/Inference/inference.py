import numpy as np

import Meshpkg.params as p
from Meshpkg.Env import State as get_state
from Meshpkg.Env.Step import step_class
from Meshpkg.Env.Reward import get_reward
from Meshpkg.Env.Action import get_action
import tensorflow as tf

def inference_step(model, episode = None):

        txt_file = open("Inference_action_record.txt", 'a')
        txt_file.write(f'\n \n Inference (Episode: {episode}) \n \n')
        
        Q_file = open("Inference_Q_record.txt", 'a')
        Q_file.write(f'\n \n Inference Q at 0th node (Episode: {episode}) \n \n')
        reward_inf_mean = 0
        s = step_class()
        for step in range(1, p.num_layer + 1):
            # epsilon greedy policy -> epsilon = 0 으로 대입
            Q_, actions = get_action(model, s.volume_mesh, 0) 
            
            txt_file.write(f'step: {step} \n actions:\n    {actions} \n')
            Q_file.write(f'step: {step} \n')
            for j in range(len(Q_)):
                Q_file.write(f'Q-value at node{j}:\n {Q_[j]}\n')
                Q_file.write(f'max Q index for node{j}: {tf.argmax(Q_[j])} \nmin Q index for node{j}: {tf.argmin(Q_[j])} \n\n')
            
            next_state_inf, reward_inf, dones_inf, info_inf, step_inf = s.step_func(actions, step)
            reward_inf_mean += np.average(reward_inf)
            
            if any(dones_inf) == 1:
                break

        txt_file.close()
        Q_file.close()
        volume_mesh_inf = s.volume_mesh

        return volume_mesh_inf, reward_inf_mean
