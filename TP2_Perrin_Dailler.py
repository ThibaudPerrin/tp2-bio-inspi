# -*- coding: utf-8 -*-
import argparse
import sys

import gym
from gym import wrappers, logger
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import random

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space
        self.size = 100000 # Memory size
        self.memory = []

    def act(self, observation, reward, done):
        return self.action_space.sample()

    def remember(self, value):
        self.memory.append(value)
        if len(self.memory) > self.size:
            self.memory.pop(0)

    def showMemory(self):
        print(self.memory)

    def getMemory(self):
        return self.memory

# class Memory():
#     def __init__(self, size):
#         self.listFifo = []
#         self.maxSize = size
#         self.counter = 0
#     def remember(self, value):
#         if len(self.listFifo) < self.maxSize and self.counter < self.maxSize:
#             self.listFifo.append(value)
#         else :
#             self.listFifo[self.counter%self.maxSize] = value
#         self.counter+=1
#     def show(self):
#         print(self.listFifo)
#     def getAll(self):
#         return self.listFifo

# class MultipleLayer(torch.nn.Module):
#     def __init__(self, D_in, H, D_out , nbcouche):
#         super(MultipleLayer, self).__init__()
#         self.n_couche = nbcouche
#         self.linear1 = torch.nn.Linear(D_in, H)
#         self.w = [torch.nn.Linear(H,H) for i in range(nbcouche)]
#         self.linear2 = torch.nn.Linear(H, D_out)

#     def forward(self, x):
#         y_pred = F.sigmoid(self.linear1(x))
#         for n in range(self.n_couche-1):
#             y_pred = F.sigmoid(self.w[n](y_pred))
#         y_pred = self.linear2(y_pred)
#         return y_pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    print(logger)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)
    listSomme = []
    episode_count = 100
    reward = 1
    done = False
    # memory = Memory(100000)
    batch_size = 32

    # model = MultipleLayer(4, 100, 2, 2)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    ######################
    ######################
    ######################
    # loss_fn = torch.nn.MSELoss(reduction='sum')
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # for t in range(nombre_iterations):
    #     for index, (image, label) in enumerate(train_loader) :
    #         counter+=1
    #         if index % 10 == 0: # evite la surcharge de message du notebook afin de ne pas avoir un message d'erreur
    #             progress = (counter * 100 / (nombre_iterations*len(train_loader)))
    #             print("Progress {:0.2f}%".format(progress), end="\r")
            
    #         # w_2c = w_2c.permute(1,0)
    #         x = image
    #         y_pred = model(x)

    #         # Compute and print loss
    #         loss = loss_fn(y_pred, label)

    #         # Zero gradients, perform a backward pass, and update the weights.
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step() 

 # target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
    for i in range(episode_count):
        somme = 0
        etat_suivant = env.reset()
        etat = etat_suivant
        while True:
            action = agent.act(etat_suivant, reward, done)
            etat_suivant, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            tensorAdd = [(etat,action,etat_suivant, reward,1. if done else 0. )]
            etat = etat_suivant
            agent.remember(tensorAdd)
            somme+= reward
            if done:
                break
            # if len(agent.getMemory()) > batch_size:
                

            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        listSomme.append(somme)
    # Close the env and write monitor result info to disk
    # agent.showMemory()
    minibatch = random.sample(agent.getMemory(), batch_size)

    x = np.arange(episode_count) 
    y = np.array(listSomme)
    plt.plot(x, y, "-ob", markersize=2, label="nom de la courbe")
    plt.show()

    env.close()









