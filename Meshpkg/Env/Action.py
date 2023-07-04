import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import tensorflow as tf


import Meshpkg.params as p
from Meshpkg.Env import State as get_state
from Meshpkg.Env.Step import step_class
from Meshpkg.Agent.policy import softmax_policy
from Meshpkg.Agent.policy import epsilon_greedy_policy

def get_action_neighbor_batch(action_batch):
    """x_action은 (batch_size*surf_length,)인 list"""
    action_neighbor_list = [ ]
    
    action_batch = np.array(action_batch).reshape(p.batch_size, p.surf_length)
    
    for i in range(p.batch_size):
        for j in range(p.surf_length):
            ###### action neighbor input 형태 설정하는 부분 #########
            if p.act_shape == 0: # [1,2]
                action_neighbor_list.append( tf.reshape(
                    [int(action_batch[i][j - 1]) / p.n_actions,
                     int(action_batch[i][(j + 1) % p.surf_length]) / p.n_actions], [1,-1]) ) 
                
            elif p.act_shape == 1: # [1, 625]
                action_neighbor_list.append( tf.reshape(tf.matmul(
                    tf.reshape(tf.one_hot(action_batch[i][j - 1], p.n_actions), [p.n_actions, 1]), 
                    tf.reshape(tf.one_hot(action_batch[i][(j + 1) % p.surf_length], p.n_actions), [1, p.n_actions])), [1, -1]) )
                
            elif p.act_shape == 2: # [1,20]
                action_neighbor_list.append( tf.concat(
                    [tf.concat([ tf.reshape(tf.one_hot(action_batch[i][j - 1] // 5, 5), [1,-1]),
                                tf.reshape(tf.one_hot(action_batch[i][j - 1] % 5, 5), [1,-1]) ], axis = 1),
                     tf.concat([ tf.reshape(tf.one_hot(action_batch[i][(j + 1) % p.surf_length] // 5, 5), [1,-1]),
                                tf.reshape(tf.one_hot(action_batch[i][(j + 1) % p.surf_length] % 5, 5), [1,-1]) ], axis = 1)], axis = 1) )
            #########################################    

    #action shape: (p.batch_size, p.surf_length, 1) -> (p.batch_size*p.surf_length, 1)
    action_neighbor = tf.squeeze(tf.convert_to_tensor(action_neighbor_list))

    return action_neighbor

"""양 옆 action random 초기화 한 다음에 Iteration 돌려서 action 도출"""
def get_action(model, volume_mesh, epsilon = None, num_iter = p.num_iter):
    "일단 action random 하게 초기화"
    action = tf.convert_to_tensor(np.random.randint(p.n_actions, size = p.surf_length))
    "state 가져오기"
    state_raw = get_state.layer_to_state(volume_mesh[-1]) # (34, 11, 2)
    state = get_state.get_new_state_2(np.array(state_raw)) # (34, 19)
    for m in range(num_iter):
        state_input_list = [ ]
        action_neighbor_list = [ ]
        with open("action_neighbor_record.txt", 'a') as txt_file:
            for j in range(p.surf_length): 
                state_input_list.append(tf.reshape(tf.convert_to_tensor(state[j]), [1, -1])) # (1, 18)
                ####### action neighbor 변경부분 #########
                if p.act_shape == 0: # [1,2]
                    action_neighbor_list.append( tf.reshape(
                        [int(action[j - 1]) / p.n_actions,
                         int(action[(j + 1) % p.surf_length]) / p.n_actions], [1, -1]) )
                
                elif p.act_shape == 1: # [1, 625]
                    action_neighbor_list.append( tf.reshape(tf.matmul(
                        tf.reshape(tf.one_hot(action[j - 1], p.n_actions), [p.n_actions, 1]), 
                        tf.reshape(tf.one_hot(action[(j + 1) % p.surf_length], p.n_actions), [1, p.n_actions])), [1, -1]) )
                
                elif p.act_shape == 2: # [1,20]
                    action_neighbor_list.append(tf.concat(
                        [tf.concat([ tf.reshape(tf.one_hot(action[j - 1] // 5, 5), [1,-1]),
                                    tf.reshape(tf.one_hot(action[j - 1] % 5, 5), [1,-1]) ], axis = 1),
                         tf.concat([ tf.reshape(tf.one_hot(action[(j + 1) % p.surf_length] // 5, 5), [1,-1]),
                                    tf.reshape(tf.one_hot(action[(j + 1) % p.surf_length] % 5, 5), [1,-1]) ], axis = 1)], axis = 1))
                #########################################
                if j == 0:
                    txt_file.write(f'\n \n --------------------Iteration [{m+1}]------------------------ \n \n')
                txt_file.write(f'node{j}: ({action[j - 1]})->({action[j]})<-({action[(j + 1) % p.surf_length]}) ')
            
        state_input = tf.squeeze(tf.convert_to_tensor(state_input_list))
        action_neighbor = tf.squeeze(tf.convert_to_tensor(action_neighbor_list))
        Q_= model([state_input, action_neighbor]) 
        action = tf.argmax(Q_, axis=1)  
        if m == num_iter - 1 and p.policy == 0:
            action =  epsilon_greedy_policy(Q_, epsilon)
        elif m == num_iter - 1 and p.policy ==1: 
            action =  softmax_policy(Q_)
    
    return Q_, action

def get_next_action(model, model_target, next_state_new, num_iter = p.num_iter):

    "next action random 하게 초기화" 
    next_action = np.random.randint(p.n_actions, size = (p.batch_size, p.surf_length)) # (batch_size, surf_length, 1)
    "next Q로 쓸 table 초기화"
    target_next_Q_all = np.zeros( (p.batch_size, p.surf_length, p.n_actions) )
    "next state 가져오기"
    next_state = next_state_new.reshape(p.batch_size, p.surf_length, -1) # (batch_size, surf_length, 19)
    
    for i in range(p.batch_size):
        for m in range(num_iter):
            next_state_input_list = [ ]
            next_action_neighbor_list = [ ]
            for j in range(p.surf_length):

                "state, 좌/우 action 넣어서 Q value 추정"
                next_state_input_list.append(tf.reshape(tf.convert_to_tensor(next_state[i][j]), [1, -1])) # (18,) --> (1, 18)
                ####### next action neighbor 변경부분 #########
                if p.act_shape == 0: # [1, 2]
                    next_action_neighbor_list.append( tf.reshape(
                        [int(next_action[i][j - 1]) / p.n_actions,
                         int(next_action[i][(j + 1) % p.surf_length]) / p.n_actions], [1,-1]) ) 
                
                elif p.act_shape == 1: # [1, 625]
                    next_action_neighbor_list.append( tf.reshape(tf.matmul(
                        tf.reshape(tf.one_hot(next_action[i][j - 1], p.n_actions), [p.n_actions, 1]), 
                        tf.reshape(tf.one_hot(next_action[i][(j + 1) % p.surf_length], p.n_actions), [1, p.n_actions])), [1, -1]) )
                
                elif p.act_shape == 2: # [1, 20]
                    next_action_neighbor_list.append( tf.concat(
                        [tf.concat([ tf.reshape(tf.one_hot(next_action[i][j - 1] // 5, 5), [1,-1]),
                                    tf.reshape(tf.one_hot(next_action[i][j - 1] % 5, 5), [1,-1]) ], axis = 1),
                        tf.concat([ tf.reshape(tf.one_hot(next_action[i][(j + 1) % p.surf_length] // 5, 5), [1,-1]),
                                    tf.reshape(tf.one_hot(next_action[i][(j + 1) % p.surf_length] % 5, 5), [1,-1]) ], axis = 1)], axis = 1) )
                
                ###############################################
                if i == 0:
                    with open("next_action_neighbor_record.txt", 'a') as txt_file:
                        if j == 0:
                            txt_file.write(f'\n \n --------------------Iteration [{m+1}]------------------------ \n \n')
                        txt_file.write(f'node{j}: ({next_action[i][j-1]})->({next_action[i][j]})<-({next_action[i][(j + 1) % p.surf_length]})')
                    
            next_state_input = tf.squeeze(tf.convert_to_tensor(next_state_input_list))
            next_action_neighbor = tf.squeeze(tf.convert_to_tensor(next_action_neighbor_list))
            
            next_Q = model([next_state_input, next_action_neighbor]) # model prediction : (1, 25)
                
            next_action[i, :] = np.array(tf.argmax(next_Q, axis=1))  #action 선택은 기존 model로
                
            if m == num_iter - 1:
                target_next_Q_all[i,:] = np.array(model_target([next_state_input, next_action_neighbor])) # Target Q 계산은 target model로
                  

    # next Q_all : (32, 34, 25) next_action : (32, 34) -> next Q_all : (32*34, 25) next_action : (32*34)
    target_next_Q_all =tf.reshape(tf.convert_to_tensor(target_next_Q_all), [p.batch_size*p.surf_length, -1] )
    next_action = tf.reshape(tf.convert_to_tensor(next_action), [p.batch_size*p.surf_length, ] )

    return target_next_Q_all, next_action  
