from __future__ import absolute_import, division, print_function, unicode_literals

import os
import datetime as dt
import torch

import colored_traceback.always

from runner import Runner
import util
import options


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from queue import Queue
import pickle
import os
import torch
import pandas as pd
from MDP.mountain_car_optimal import trajectory
  

class QLearning:
    def __init__(self, actions_space, learning_rate=0.01, reward_decay=0.99, e_greedy=0.6):
        self.actions = actions_space    
        #self.target                     
        self.lr = learning_rate         
        self.gamma = reward_decay       
        self.epsilon = e_greedy         
        self.num_pos = 20               
        self.num_vel = 14               
        self.q_table =  np.random.uniform(low=-1, high=1, size=(self.num_pos*self.num_vel, self.actions.n)) 
        self.pos_bins = self.toBins(-1.2 , 0.6 , self.num_pos)
        self.vel_bins = self.toBins(-0.07 , 0.07, self.num_vel)

    def choose_action(self,state):
        if np.random.uniform() < self.epsilon:
            action = np.argmax(self.q_table[state])
        else:
            action = self.actions.sample()
        return action
    
    def randpi(self, state, tau):
        Q_vec = self.q_table[state]
        src = torch.from_numpy(Q_vec)
        try:
            p = torch.nn.functional.softmax(src/tau, dim=1)
        except:
            p = torch.from_numpy(np.array([0.33,0.33,0.33]))
        action = torch.multinomial(p,1,replacement = True)
        return action.to(torch.int64)

    def randpis(self, states, tau):
        Q_vec = self.q_table[states]
        src = torch.from_numpy(Q_vec)
        try:
            p = torch.nn.functional.softmax(src/tau, dim=1)
        except:
            p = torch.from_numpy(np.array([0.33,0.33,0.33]))
        action = torch.multinomial(p,1,replacement = True)
        return int(action) 
        

    def toBins(self,clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num + 1)
   
    def digit(self,x, bin):
        n = np.digitize(x,bins = bin)
        if x== bin[-1]:
            n=n-1
        return n
    
    def digits(self, xs, bin):
        n = np.digitize(xs.cpu(), bins = bin)
        n[xs.cpu() >= bin[-1]] = n[xs.cpu() >= bin[-1]] - 1
        return n

    def digitize_state(self,observation):
        if observation[1] != {}:
            cart_pos, cart_v = observation
        else:
            cart_pos, cart_v = observation[0]
        digitized = [self.digit(cart_pos,self.pos_bins),
                    self.digit(cart_v,self.vel_bins),]
        return (digitized[1]-1)*self.num_pos + digitized[0]-1
        
    def digitize_states(self, observations):
        cart_poses = observations[:,0]
        cart_vs = observations[:,1]

        digitized = [self.digits(cart_poses,self.pos_bins),
                    self.digits(cart_vs,self.vel_bins),]
        
        return (digitized[1]-1)*self.num_pos + digitized[0]-1

    def learn(self, state, action, r, next_state):
        next_action = np.argmax(self.q_table[next_state]) 
        q_predict = self.q_table[state, action]
        q_target = r + self.gamma * self.q_table[next_state, next_action]   
        self.q_table[state, action] += self.lr * (q_target - q_predict)     
    
    
    def pdf(self, actions, states, tau):
        Q_vec = self.q_table[states]
        src = torch.from_numpy(Q_vec)
        ps = torch.nn.functional.softmax(src/tau, dim=1)
        p = ps[np.arange(len(actions)), actions].numpy()
        return p

print(util.yellow("="*80))
print(util.yellow(f"\t\tTraining start at {dt.datetime.now().strftime('%m_%d_%Y_%H%M%S')}"))
print(util.yellow("="*80))
print(util.magenta("setting configurations..."))
opt = options.set()

def main(opt):
    # run = Runner(opt)
    # run.sb_alternate_imputation_train_mdp(opt)
    with open(os.getcwd()+'/q_learning_model/carmountain.model_original_version', 'rb') as f:
            print(os.getcwd())
            agent = pickle.load(f)
            np.random.seed(1)
    s0_sampler = trajectory()
    size = 50
    tau = 1
    s0,s_m,a_m,r_m = s0_sampler.sample_trajectory(100, size, tau)
    print(r_m.sum()/size)
    horizon = 100
    states = s0
    reward_sum = 0
    print(r_m)
    # for t in np.arange(horizon):
    #     # plt.scatter(s_m[:,t,0], s_m[:,t,1])
    #     # plt.savefig(str(t) + '.png')

    #     obs = torch.from_numpy(states).to(opt.device)
    #     states = agent.digitize_states(obs)
    #     actions = agent.randpi(states, tau).to(opt.device)
    #     zeros_target = torch.zeros((size, 3)).to(opt.device)
    #     x_cond = torch.cat((obs, actions), dim = 1)
    #     x_cond = torch.cat((x_cond, zeros_target), dim = 1)
    #     x_cond = x_cond.unsqueeze(1)
    #     cond_mask = torch.zeros_like(x_cond).to(opt.device)
    #     cond_mask[:,:,:3] = 1
    #     cond_mask = cond_mask.unsqueeze(1).float()
    #     x_cond = x_cond.unsqueeze(1).float()
    #     target = run.conditional_sampler(opt, x_cond, cond_mask)
    #     rewards = target[:,-1]
    #     states = target[:,3:5].cpu().numpy()
    #     plt.scatter(states[:,0], states[:,1],label = 'CSDI')
    #     plt.scatter(s_m[:,t,0], s_m[:,t,1],label = 'ground_truth')
    #     plt.legend()
    #     plt.savefig('./image/'+ str(t) + '.png')
    #     plt.clf()
    #     print(rewards)
    #     reward_sum += torch.sum(rewards).cpu().numpy()
    # print(reward_sum/size)

if not opt.cpu:
    with torch.cuda.device(opt.gpu):
        main(opt)
else: main(opt)
