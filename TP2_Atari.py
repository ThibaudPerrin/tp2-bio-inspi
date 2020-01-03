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
from gym.spaces import Box
from gym.wrappers import TimeLimit


class AtariPreprocessing(gym.Wrapper):
    r"""Atari 2600 preprocessings.

    * NoopReset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost: turned off by default. Not recommended by Machado et al. (2018).
    * Resize to a square image: 84x84 by default
    * Grayscale observation: optional
    * Scale observation: optional
    Args:
        env (Env): environment
        noop_max (int): max number of no-ops
        frame_skip (int): the frequency at which the agent experiences the game.
        screen_size (int): resize Atari frame
        terminal_on_life_loss (bool): if True, then step() returns done=True whenever a
            life is lost.
        grayscale_obs (bool): if True, then gray scale observation is returned, otherwise, RGB observation
            is returned.
        scale_obs (bool): if True, then observation normalized in range [0,1] is returned. It also limits memory
            optimization benefits of FrameStack Wrapper.
    """

    def __init__(self, env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True,
                 scale_obs=False):
        super().__init__(env)
        assert frame_skip > 0
        assert screen_size > 0
        assert noop_max >= 0
        if frame_skip > 1:
            assert 'NoFrameskip' in env.spec.id, 'disable frame-skipping in the original env. for more than one' \
                                                 ' frame-skip as it will be done by the wrapper'
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.terminal_on_life_loss = terminal_on_life_loss
        self.grayscale_obs = grayscale_obs
        self.scale_obs = scale_obs

        # buffer of most recent four observations for max pooling
        print(env.observation_space)
        if grayscale_obs:
            self.obs_buffer = [np.empty(env.observation_space.shape[:2], dtype=np.uint8),
                               np.empty(env.observation_space.shape[:2], dtype=np.uint8),
                               np.empty(env.observation_space.shape[:2], dtype=np.uint8),
                               np.empty(env.observation_space.shape[:2], dtype=np.uint8)]
        else:
            self.obs_buffer = [np.empty(env.observation_space.shape, dtype=np.uint8),
                               np.empty(env.observation_space.shape, dtype=np.uint8),
                               np.empty(env.observation_space.shape, dtype=np.uint8),
                               np.empty(env.observation_space.shape, dtype=np.uint8)]

        self.ale = env.unwrapped.ale
        self.lives = 0
        self.game_over = False

        _low, _high, _obs_dtype = (0, 255, np.uint8) if not scale_obs else (0, 1, np.float32)
        if grayscale_obs:
            self.observation_space = Box(low=_low, high=_high, shape=(screen_size, screen_size), dtype=_obs_dtype)
        else:
            self.observation_space = Box(low=_low, high=_high, shape=(screen_size, screen_size, 3), dtype=_obs_dtype)

    def step(self, action):
        R = 0.0
        result_array = []
        for t in range(self.frame_skip):
            _, reward, done, info = self.env.step(action)
            R += reward
            self.game_over = done

            if self.terminal_on_life_loss:
                new_lives = self.ale.lives()
                done = done or new_lives < self.lives
                self.lives = new_lives

            if done:
                break
            if t == self.frame_skip - 4:
                if self.grayscale_obs:
                    self.ale.getScreenGrayscale(self.obs_buffer[0])
                else:
                    self.ale.getScreenRGB2(self.obs_buffer[0])
            elif t == self.frame_skip - 3:
                if self.grayscale_obs:
                    self.ale.getScreenGrayscale(self.obs_buffer[1])
                else:
                    self.ale.getScreenRGB2(self.obs_buffer[1])
            elif t == self.frame_skip - 2:
                if self.grayscale_obs:
                    self.ale.getScreenGrayscale(self.obs_buffer[2])
                else:
                    self.ale.getScreenRGB2(self.obs_buffer[2])
            elif t == self.frame_skip - 1:
                if self.grayscale_obs:
                    self.ale.getScreenGrayscale(self.obs_buffer[3])
                else:
                    self.ale.getScreenRGB2(self.obs_buffer[3])

            result_array.append(self._get_obs())
        return np.array(result_array), R, done, info


    def reset(self, **kwargs):
        # NoopReset
        self.env.reset(**kwargs)
        noops = self.env.unwrapped.np_random.randint(1, self.noop_max + 1) if self.noop_max > 0 else 0
        for _ in range(noops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset(**kwargs)

        self.lives = self.ale.lives()
        if self.grayscale_obs:
            self.ale.getScreenGrayscale(self.obs_buffer[0])
        else:
            self.ale.getScreenRGB2(self.obs_buffer[0])
        self.obs_buffer[1].fill(0)
        return self._get_obs()

    def _get_obs(self):
        import cv2
        if self.frame_skip > 1:  # more efficient in-place pooling
            np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])
        obs = cv2.resize(self.obs_buffer[0], (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)

        if self.scale_obs:
            obs = np.asarray(obs, dtype=np.float32) / 255.0
        else:
            obs = np.asarray(obs, dtype=np.uint8)
        return obs


class RandomAgent(object):
    """The world's simplest agent!"""

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
        self.size = 100000  # Memory size
        self.memory = []
        self.batch_size = 64
        self.state_size = 4
        self.action_size = 2
        self.learning_rate = 1e-3
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.state_size, 100),
            torch.nn.Sigmoid(),
            torch.nn.Linear(100, self.action_size),
        )
        self.model_duplicata = torch.nn.Sequential(
            torch.nn.Linear(self.state_size, 100),
            torch.nn.Sigmoid(),
            torch.nn.Linear(100, self.action_size),
        )
        self.Tau = 0.5

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.learn_state = 0
        self.gamma = 0.99

        self.upadteModel()

    # action 1 = droite action 0 = gauche
    def act(self, observation, reward, done):
        return

    def upadteModel(self):
        return

    def remember(self, value):
        return

    def showMemory(self):
        return

    def getMemory(self):
        return self.memory

    def retry(self, batch_size):
        return


if __name__ == '__main__':
    #np.set_printoptions(threshold=sys.maxsize)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='BreakoutNoFrameskip-v4', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    # print(logger)

    env = gym.make(args.env_id)
    print(env)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4, scale_obs=True), directory=outdir, force=True)
    agent = RandomAgent(env.action_space)
    listSomme = []
    episode_count = 1
    reward = 1

    for i in range(episode_count):
        somme = 0
        etat = env.reset()
        done, step_i = False, 0

        while step_i < 50:
            env.render()

            etat_suivant, reward , done, _ = env.step(env.action_space.sample())
            reward = reward if not done else -10
            step_i += 1
        i = 1

    env.close()
