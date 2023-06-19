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


    def State_Reward_plot(self, plot_dict, episode = None):
        reward_list = plot_dict['reward']
        self.createFolder('State_Reward')
        plt.figure(dpi = 350)
        for i in range(len(reward_list)-1):
            plt.plot(reward_list[i][:], label=f'Layer[{i+1}]', color = self.colors[i%(self.n_actions)])
            plt.xlabel('State')
            plt.ylabel('Reward')

        plt.legend(loc=(1.0, 0), ncol = 1, fontsize = 8.9531 )
        plt.ylim([0, 1])
        plt.title(f'State - Reward (Episode: {episode+1})')
        if episode != None:
            plt.savefig(f'State_Reward/State_Reward_epi_{episode+1}.jpg', dpi = 350, bbox_inches = 'tight')
        plt.close()


    def Episode_Reward_plot(self, reward_episode, episode = None):
        episode_num = [i for i in range(1,len(reward_episode)+1)]
        self.createFolder('Episode_reward')
        plt.figure(dpi = 350)
        plt.plot(episode_num, reward_episode, color = 'black')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title('Episode - Accumulated Reward')
        if episode != None:
            plt.savefig(f'Episode_reward/Episode_Reward_epi_{episode}.jpg', dpi = 350)
        plt.clf()
        plt.close("all")
        with open(f'Episode_reward/reward_epi_{episode}.p', 'wb') as fe:    
            pickle.dump(reward_episode, fe)
        


