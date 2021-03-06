# -*- coding: utf-8 -*-
import argparse
import sys
import gym
from gym import wrappers, logger
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random
from random import choices



class RandomAgent(object):
    def __init__(self, action_space):
        """Initialize an Agent object.
        
        Params
        =======
            size (int): size of the memory
            memory (array()): memory of the agent
            batch_size (int): size of the part of memory which is selected (N)
            state_size (int): dimension of each state (D_in)
            action_size (int): dimension of each action (D_out)
        """
        
        self.action_space = action_space
        self.size = 100000 # Memory size
        self.memory = []
        self.batch_size = 32
        self.state_size = 4
        self.action_size = 2
        self.learning_rate = 1e-3
        self.model = MultipleLayer(self.state_size, 100, self.action_size, 1)
        self.model_duplicata = MultipleLayer(self.state_size, 100, self.action_size, 1)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.learn_state = 0
        self.gamma = 0.95

        self.upadteModel()


    # action 1 = droite action 0 = gauche
    def act(self, observation, reward, done):
        epsilon = 0.1
        rnd = random.uniform(0, 1)
        res = self.model(torch.tensor(observation).float())
        maxval, idx = res.max(0)
        maxval, idx2 = res.min(0)
        if rnd < 1-epsilon:
            indices = idx.item()
        else:
            indices = idx2.item()
        return indices

    def upadteModel(self):
        self.model_duplicata.linear1 = self.model.linear1
        self.model_duplicata.w = self.model.w
        self.model_duplicata.linear2 = self.model.linear2

    def remember(self, value):
        self.memory.append(value)
        if len(self.memory) > self.size:
            self.memory.pop(0)

    def showMemory(self):
        print(self.memory)

    def getMemory(self):
        return self.memory

    def retry(self, batch_size):
        minibatch = random.sample(self.memory, self.batch_size)
        for etat, action, etat_suivant, reward, done in minibatch:

            qO = self.model(torch.tensor(etat).float())

            qOsa = qO[action]

            qO_suivant = self.model_duplicata(torch.tensor(etat_suivant).float())

            rPlusMaxNext = reward + self.gamma*torch.max(qO_suivant)

            if not done :
                JO = pow(qOsa - rPlusMaxNext, 2)
            else :
                JO = pow(qOsa - reward, 2)
            loss = self.loss_fn(qOsa, JO)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            

            if (self.learn_state % 10000 == 0):
                print("learn_state : ", self.learn_state)
                self.upadteModel()

            self.learn_state +=1


class MultipleLayer(torch.nn.Module):
    def __init__(self, D_in, H, D_out, nbcouche):
        super(MultipleLayer, self).__init__()
        self.n_couche = nbcouche
        self.linear1 = torch.nn.Linear(D_in, H)
        self.w = [torch.nn.Linear(H,H) for i in range(nbcouche)]
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear1(x))
        for n in range(self.n_couche-1):
            y_pred = torch.sigmoid(self.w[n](y_pred))
        y_pred = self.linear2(y_pred)
        return y_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()

    logger.set_level(logger.INFO)


    env = gym.make(args.env_id)

    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)
    listSomme = []
    episode_count = 260
    reward = 1
    max_reward = 500

    etat_space = env.observation_space.shape[0]
    action_space = env.action_space.n


    for i in range(episode_count):
        somme = 0
        etat = env.reset()
        done = False
        
        while True:
            #  env.render()
            action = agent.act(etat, reward, done)
            etat_suivant, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            tensorAdd = (etat, action, etat_suivant, reward, done)

            agent.remember(tensorAdd)
            etat = etat_suivant

            somme += reward
            if done:
                agent.upadteModel()
                break
            if somme > max_reward:
                break

            if len(agent.memory) > agent.batch_size:
                agent.retry(agent.batch_size)

        listSomme.append(somme)

    x = np.arange(episode_count) 
    y = np.array(listSomme)
    plt.plot(x, y, "-ob", markersize=2, label="nom de la courbe")
    plt.show()
    env.close()









