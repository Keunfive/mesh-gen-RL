import numpy as np
import tensorflow as tf
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pickle
import Meshpkg.params as p

class graph_plot:

    def __init__(self, seed = 42, actions = [i for i in range(1,6)], n_savefig = 0):
        self.colors = ['black','deepskyblue','sandybrown', 'darkgreen', 'm',
                        'gold','darkmagenta','slateblue', 'blue','rosybrown',
                        'gray','red','purple','turquoise','darkorange', 
                        'lightseagreen','chocolate','crimson', 'lightslategray','blueviolet',
                        'fuchsia', 'darkolivegreen', 'palegoldenrod', 'burlywood', 'cornflowerblue']

        self.n_actions = p.n_actions

    def createFolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.mkdir(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)


    def Episode_Reward_train_plot(self, reward_episode, episode = None):
        episode_num = [i for i in range(1,len(reward_episode)+1)]
        self.createFolder('Episode_reward_train')
        plt.figure(dpi = 350)
        plt.plot(episode_num, reward_episode, color = 'black')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title('Episode - Accumulated Reward')
        if episode != None:
            plt.savefig(f'Episode_reward_train/Episode_Reward_epi_{episode}.jpg', dpi = 350)
        plt.clf()
        plt.close("all")
        with open(f'Episode_reward_train/reward_epi_{episode}.p', 'wb') as fe:    
            pickle.dump(reward_episode, fe)
        

    def Episode_Reward_inf_plot(self, reward_inf_list, episode = None):
        episode_num = [i for i in range(5,5*len(reward_inf_list)+1,5)]
        self.createFolder('Episode_reward_inf')
        plt.figure(dpi = 350)
        plt.plot(episode_num, reward_inf_list, color = 'black')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title('Episode - Accumulated Reward(Inference)')
        if episode != None:
            plt.savefig(f'Episode_reward_inf/Episode_Reward_epi_{episode}.jpg', dpi = 350)
        plt.clf()
        plt.close("all")
        with open(f'Episode_reward_inf/reward_epi_{episode}.p', 'wb') as fe:    
            pickle.dump(reward_inf_list, fe)
