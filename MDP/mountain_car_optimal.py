import numpy as np
import pandas as pd
import time
import gymnasium as gym
import pickle
import torch
from gymnasium.wrappers import TransformObservation
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.classic_control import MountainCarEnv
import os

class ExtendedMountainCarEnv(MountainCarEnv):
    def reset(self,seed):
        np.random.seed(seed)
        pos = np.random.uniform(-0.6, -0.4,1)
        vel = np.random.uniform(-0.01, 0.01,1)
        self.state = np.array((pos, vel))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)


class trajectory(object):
    def __init__(self, env_name = 'MountainCar-v0') -> None:
        max_episode_steps = 200
        env = ExtendedMountainCarEnv()
        self.env = TimeLimit(env, max_episode_steps)  
        with open(os.getcwd()+'/q_learning_model/carmountain.model_original_version', 'rb') as f:
            print(os.getcwd())
            self.agent = pickle.load(f)
        self.agent.actions = self.env.action_space    
        self.agent.epsilon = 1
        self.token = 0
        np.random.seed(1)
        
    def sample_one_trajectory(self, horizon,tau, seed):
        s0 = self.env.reset(seed = seed)
        self.token += 1
        states = np.zeros((horizon,2))
        actions = np.zeros((horizon))
        rewards = np.zeros((horizon))
        state = self.agent.digitize_state(s0)
        for t in range(horizon):   
            action = self.agent.randpi(state, tau).numpy()
            observation, reward, done,_, info = self.env.step(int(action[0]))
            states[t] = observation.reshape(-1,2)
            actions[t] = int(action[0])
            if done:
                rewards[t] = 0
            else:
                rewards[t] = reward
            # if done:
            #     rewards[t] = 1
            # elif observation[0] > 0.2:
            #     rewards[t] = 0.5
            # elif observation[0] > 0:
            #     rewards[t] = 0
            # else:
            #     rewards[t] = reward
            next_state = self.agent.digitize_state(observation)
            state = next_state
        return s0.reshape(1,-1), states, actions, rewards
    
    def density(self, states, actions, tau):
        states = np.apply_along_axis(self.agent.digitize_state, arr = states, axis = 1)
        return self.agent.pdf(actions, states, tau)
    
    def sample_trajectory(self, horizon, size, tau,seed = 0):
        # sample continuous state
        np.random.seed(seed)
        torch.manual_seed(seed)
        seeds = np.random.randint(10000, size = size)
        s_m = np.zeros((size, horizon,2))
        a_m = np.zeros((size, horizon))
        r_m = np.zeros((size, horizon))
        s_0= np.zeros((size, 2))
        for i in range(size):
            s_0[i], s_m[i], a_m[i], r_m[i] = self.sample_one_trajectory(horizon, tau, seeds[i])
        # s_m[]
        s_m[:,:,1] = s_m[:,:,1] * 25 / 2
        s_0[:,1] = s_0[:,1] * 25 / 2
        s_m[:,:,1] = s_m[:,:,1] * 5
        s_m[:,:,0] = (s_m[:,:,0]  + 0.5)* 5
        return s_0, s_m, a_m, r_m
    
    
    
