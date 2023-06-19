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

def training_step_mean_DDQN(model, model_target, replay_memory):

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
    next_state_new = get_state.get_new_state_1s(next_state)
    next_Q, next_action = get_next_action(model, model_target, next_state_new)
    next_mask = tf.cast(tf.one_hot(next_action, p.n_actions), tf.float64)
    max_next_Q = tf.reduce_sum(next_Q * next_mask, axis=1, keepdims=False)

    target_Q_values = reward + (1 - done) * p.discount_rate * max_next_Q
    
    target_file = open("target_Q_record.txt", 'a')
    
    target_file.write(f'\n\n---------Target Q for {0}th batch-------------\n\n')
    for j in range(p.surf_length):
        target_file.write(f'node{j}: {target_Q_values[0*p.batch_size +j]:.3f} ')
    target_file.close()
    

    """Q value 예측 [DDQN]"""
    "state normalization"
    state_new = tf.convert_to_tensor(get_state.get_new_state_1s(state))
    "neighborhood actions"
    action_neighbor = get_action_neighbor_batch(action)
    "action masking"
    action = tf.convert_to_tensor(action)
    mask = tf.one_hot(action, p.n_actions)


    with tf.GradientTape() as tape:
        
        all_Q_values = model([state_new, action_neighbor])
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss_mean = tf.reduce_mean(p.loss_fn(target_Q_values, Q_values))

    grads = tape.gradient(loss_mean, model.trainable_variables)
    # gradients = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads]

    p.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss_mean, state_new, next_state_new, Q_values, target_Q_values

