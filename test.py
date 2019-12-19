import argparse
import sys

import gym
from gym import wrappers, logger
import matplotlib.pyplot as plt
import torch

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

class ListFifo():
    def __init__(self, size):
        self.listFifo = []
        self.maxSize = size
        self.counter = 0
    def add(self, value):
        if len(self.listFifo) < self.maxSize and self.counter < self.maxSize:
            self.listFifo.append(value)
        else :
            self.listFifo[self.counter%self.maxSize] = value
        self.counter+=1
    def show(self):
        print(self.listFifo)

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
    listFifo = ListFifo(100000)
    # for i in range(12):
    #     tensorAdd = torch.tensor([[[i],[i]]],dtype=float)
    #     # tensorAdd = tensorAdd.permute(1,0)
    #     listFifo.add(tensorAdd)


    for i in range(episode_count):
        somme = 0
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            tensorAdd = [[[ob,[reward],ob,[1. if done else 0. ]]]]
            listFifo.add(tensorAdd)
            somme+= reward
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        listSomme.append(somme)
    # Close the env and write monitor result info to disk
    listFifo.show()
    

    x = np.arange(episode_count) 
    y = np.array(listSomme)
    plt.plot(x, y, "-ob", markersize=2, label="nom de la courbe")
    plt.show()


    env.close()