import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import tensorflow as tf
import itertools
import time
from Meshpkg import params as p
from Meshpkg.Env import State as get_state
from Meshpkg.Env.Action import get_action_neighbor_batch
from Meshpkg.Env.Action import get_next_action


    
""" 
점 전체를 한번에 신경망 Weight update 하는 함수 [Double DQN] 
"""

def training_step_mean_DDQN(model, model_target, replay_memory, episode):

    "replay memory에서 batch를 random sampling"
    indices = np.random.randint(len(replay_memory), size = p.batch_size) # 
    "replay memory에서 batch를 순차적으로 sampling"
    # indices=[i for i in range(self.batch_size)] 

    batch = [replay_memory[index] for index in indices]
    state, action, reward, next_state, done, step = [ [experience[field_index] for experience in batch] for field_index in range(6)]

    "list size 조정: (batch_size, length ) -> (batch_size*length, )"
    list_flatten = lambda x, opt = "np": np.array(list(itertools.chain(*x))) if opt=="np" else list(itertools.chain(*x))
    state, action, reward, next_state, done = list_flatten(state), list_flatten(action, opt="list"), list_flatten(reward), list_flatten(next_state), list_flatten(done)
    

    """Target Q value 계산 [DDQN]"""
    "next state normalization"
    next_state_new = get_state.get_new_state_2(next_state)
    next_Q, next_action = get_next_action(model, model_target, next_state_new)
    next_mask = tf.cast(tf.one_hot(next_action, p.n_actions), tf.float64)
    max_next_Q = tf.reduce_sum(next_Q * next_mask, axis=1, keepdims=False)

    target_Q_values = reward + (1 - done) * p.discount_rate * max_next_Q
    
    target_file = open("target_Q_record.txt", 'a')
    
    target_file.write(f'\n\n---------Target Q for {0}th batch [episode:{episode}]-------------\n\n')
    for j in range(p.surf_length):
        target_file.write(f' Target Q(node{j}): {target_Q_values[0*p.batch_size +j]:.3f} \n')
    target_file.close()
    

    """Q value 예측 [DDQN]"""
    "state normalization"
    state_new = tf.convert_to_tensor(get_state.get_new_state_2(state))
    "neighborhood actions"
    action_neighbor = get_action_neighbor_batch(action)
    "action masking"
    action = tf.convert_to_tensor(action)
    mask = tf.one_hot(action, p.n_actions)


    with tf.GradientTape() as tape:
        
        all_Q_values = model([state_new, action_neighbor])
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(p.loss_fn(target_Q_values, Q_values))

    loss_file = open("Train_loss_record.txt", 'a')
    loss_file.write(f'\n\n---------Training Loss [episode:{episode}]-------------\n\n')
    loss_file.write(f' Loss(node mean): {np.array(loss):.3f} \n')
    loss_file.close()
    
    grads = tape.gradient(loss, model.trainable_variables)
    # gradients = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads]
    p.optimizer.apply_gradients(zip(grads, model.trainable_variables))



""" 
점(Agent) 한개씩 신경망 Weight update 하는 함수 [Double DQN] 
"""

def training_step_each_DDQN(model, model_target, replay_memory, episode):

    "replay memory에서 batch를 random sampling"
    indices = np.random.randint(len(replay_memory), size = p.batch_size) # 
    "replay memory에서 batch를 순차적으로 sampling"
    # indices=[i for i in range(self.batch_size)] 

    batch = [replay_memory[index] for index in indices]
    state, action, reward, next_state, done, step = [ [experience[field_index] for experience in batch] for field_index in range(6)]

    "list size 조정: (batch_size, length ) -> (batch_size*length, )"
    list_flatten = lambda x, opt = "np": np.array(list(itertools.chain(*x))) if opt=="np" else list(itertools.chain(*x))
    state, action, reward, next_state, done = list_flatten(state), list_flatten(action, opt="list"), list_flatten(reward), list_flatten(next_state), list_flatten(done)
    

    """Target Q value 계산 [DDQN]"""
    "next state normalization"
    next_state_new = get_state.get_new_state_2(next_state)
    next_Q, next_action = get_next_action(model, model_target, next_state_new)
    next_mask = tf.cast(tf.one_hot(next_action, p.n_actions), tf.float64)
    max_next_Q = tf.reduce_sum(next_Q * next_mask, axis=1, keepdims=False)

    target_Q_values = reward + (1 - done) * p.discount_rate * max_next_Q
    
    target_file = open("target_Q_record.txt", 'a')
    target_file.write(f'\n\n---------Target Q for {0}th batch [episode:{episode}]-------------\n\n')
    for j in range(p.surf_length):
        target_file.write(f' Target Q(node{j}): {target_Q_values[0*p.batch_size +j]:.3f} \n')
    target_file.close()
    

    """현재 상태(S_t_i), 주변 Action(A_t_(i+_1))을 통한 Q value estimation"""
    "state normalization"
    state_new = get_state.get_new_state_2(state)
    "neighborhood actions"
    action_neighbor = get_action_neighbor_batch(action)
    "action masking"
    action = tf.convert_to_tensor(action)
    ########################################################
    loss = [] 
    Q_values = []
    
    loss_file = open("Train_loss_record.txt", 'a')
    loss_file.write(f'\n\n---------Training Loss [episode:{episode}]-------------\n\n')
    
    for i in range(p.surf_length):
        
        state_sorted = [ ]
        action_neighbor_sorted = [ ]
        action_sorted = [ ]
        target_Q_sorted = [ ]

        for j in range(p.batch_size):
            state_sorted.append(state_new[i + j*(p.surf_length)])
            action_neighbor_sorted.append(action_neighbor[i + j*(p.surf_length)])
            action_sorted.append(action[i + j*(p.surf_length)])
            target_Q_sorted.append(target_Q_values[i + j*(p.surf_length)])
        
        state_sorted = tf.convert_to_tensor(state_sorted)
        action_neighbor_sorted = tf.convert_to_tensor(action_neighbor_sorted)
        action_sorted = tf.convert_to_tensor(action_sorted)
        mask_sorted = tf.one_hot(action_sorted, p.n_actions)
        target_Q_sorted = tf.convert_to_tensor(target_Q_sorted)

        with tf.GradientTape() as tape:

            all_Q_sorted = model([state_sorted, action_neighbor_sorted])                 # all_Q_values = model(state)
            Q_sorted = tf.reduce_sum(all_Q_sorted * mask_sorted, axis=1, keepdims=True)
            loss_agent = tf.reduce_mean(p.loss_fn(target_Q_sorted, Q_sorted))
        
        loss_file.write(f' Loss(node{i}): {np.array(loss_agent):.3f} \n')
        Q_values.append(Q_sorted)    
        loss.append(loss_agent)

        grads = tape.gradient(loss_agent, model.trainable_variables)
        p.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    loss_file.close()

