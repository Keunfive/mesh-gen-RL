import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import Meshpkg as mp

import pandas as pd
import numpy as np
import tensorflow as tf

import time
import datetime
import matplotlib.pyplot as plt
from tensorflow import keras
from collections import deque
from tqdm import tqdm
import pickle

"Parameter 정의"
p = mp.params

"Seed 설정"
seed = 42
mp.Initialize.my_seed.my_seed_everywhere(42)

"Episode 수"
start_episode = 100
n_episodes = 200

"model, target model(Double DQN) 정의"

model = tf.keras.models.load_model(f'model_storage/DDQN_spline_1_episode_{start_episode}')

model_target = keras.models.clone_model(model)
model_target.set_weights(model.get_weights())

print(model.summary())

"Replay_memory 정의"
with open(f'replay_memory/replay_memory_{start_episode}.p', 'rb') as fr:  
    replay_memory = pickle.load(fr)

"Inference 주기"
episode_inference = 5

"Neural Network model 저장 주기, 저장 여부"
episode_save = 50
save_model = True

"Episode - reward list/ Time initialize"
with open(f'Episode_loss_reward/reward_epi_{start_episode}.p', 'rb') as fe:     
    reward_list = pickle.load(fe)
    
start = time.time()

for episode in range(start_episode+1, n_episodes+1): 
    
    s = mp.Env.Step.step_class()
    state = s.reset()
    step_ended = 0
    # step_bar = tqdm(range(1, p.num_layer+1), desc = f'< Episode: {episode} > Steps' , leave = True, maxinterval = 0.1, position = 1)
    reward_episode = 0
    epsilon = max(((p.epsilon_start)**episode), p.epsilon_min) # epsilon 0.01 도달 까지 4603 필요
    for step in range(1, p.num_layer+1):
        _, actions = mp.Env.Action.get_action(model, s.volume_mesh, epsilon)
        next_state, reward, done, info, steps =  s.step_func(actions, step, episode)
        replay_memory.append((state, actions, reward, next_state, done, steps))
        state = next_state
        reward_episode += np.average(reward)
        if any(done) == 1:
            step_ended = step
            reward_list.append(reward_episode)
            
            with open("episode_step_record.txt", 'a') as epistep_file:
                epistep_file.write(f' \n<episode: {episode}> Step ended: {step_ended} ')
                if episode == start_episode+1:
                    end1 = start
                end2 = time.time()
                epi_time = str(datetime.timedelta(seconds= (end2 - end1)))
                short1 = epi_time.split(".")[0]
                total_time = str(datetime.timedelta(seconds= (end2 - start)))
                short2 = total_time.split(".")[0]
                epistep_file.write(f"  Time per episode: {short1} (Total: {short2})\n") # epi 시간, 누적시간 출력
                end1 = end2
                
            if step_ended != p.num_layer:
                replay_memory = mp.Train.replay_penalty.penalty_reward(replay_memory, info, step_ended, 5, 3)
            break
    # step_bar.close()
    "replay memory 다 차면, episode 끝나고 model training 시작"
    if len(replay_memory) == p.buffer_size:
        loss_mean, state_new, next_state_new, Q_values, target_Q_values = mp.Train.model_training.training_step_mean_DDQN(model, model_target, replay_memory)
        
    "episode (episode_inference)회마다 Inference"
    if episode % (episode_inference) == 0:
        volume_mesh_inf = mp.Inference.inference.inference_step(model, episode)
        mp.Inference.render.render(volume_mesh_inf, episode)

    "episode (episode_target)회마다 Target model update"
    if episode % (p.episode_target) == 0:
        model_target.set_weights(model.get_weights())

    "episode (episode_save)회마다 model, replay memory, episode-reward 저장"
    if (episode % (episode_save) == 0) and (save_model):
        model.save(f'model_storage/DDQN_{p.mesh_name}_episode_{episode}')
        
        mp.Inference.graph.graph_plot().createFolder('replay_memory')
        with open(f'replay_memory/replay_memory_{episode}.p', 'wb') as fr:    
            pickle.dump(replay_memory, fr)
            
        mp.Inference.graph.graph_plot().Episode_Reward_plot(reward_list, episode)


print ('Finish at: ',str(datetime.timedelta(seconds= (time.time() - start))))