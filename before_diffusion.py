
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from queue import Queue
import pickle
import pandas as pd
import time
import csv
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
        return int(action)
        

    def toBins(self,clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num + 1)
   
    def digit(self,x, bin):
        n = np.digitize(x,bins = bin)
        if x== bin[-1]:
            n=n-1
        return n

    def digitize_state(self,observation):
        if observation[1] != {}:
            cart_pos, cart_v = observation
        else:
            cart_pos, cart_v = observation[0]
        digitized = [self.digit(cart_pos,self.pos_bins),
                    self.digit(cart_v,self.vel_bins),]
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


def prepare_training_data(s0, s_m, a_m, r_m):
    horizon = s_m.shape[0]
    s_m = np.concatenate((np.expand_dims(s0, axis = 0), s_m))
    target_dim = s0.shape[1] + 1
    cond_dim = s0.shape[1] + 1
    sample_size = horizon * s0.shape[0]
    data_train = np.zeros((sample_size * horizon , (cond_dim + target_dim)))
    for t in np.arange(horizon):
        for i in np.arange(s0.shape[0]):
            data_train[i+t * s0.shape[0],] = np.concatenate((s_m[t,i],a_m[t,i].reshape(1),s_m[t+1,i],r_m[t,i].reshape(1)))
    return data_train

def main():
    # yaml_file = 'before_diffusion.yaml'
    # with open(yaml_file, encoding='UTF-8') as yaml_file:
    #     args = yaml.safe_load(yaml_file)
    
    # n_traj = args["n_traj"]
    # num_eps = args["num_eps"]
    # horizon = args["H"]
    # represent_nrow = args["represent_nrow"]
    # batch_size = args["batch_size"] 
    # uniform_domain = args["uniform_domain"]
    # eps = args["eps"]
    # tau_pi = args["tau_pi"]
    # tau_mu = args["tau_mu"]
    # reference = args["reference"]
    # sample_eps = args["sample_eps"]
    # sample_reference = args["sample_reference"]
    # after_sample_dir = args["after_sample_dir"]
    
    # generate trajectory
    tau_mu = 0.1
    horizon = 100
    n_traj = 2000
    my_traject = trajectory()
    s0, s_m_mu, a_m_mu, r_m_mu = my_traject.sample_trajectory(horizon, n_traj, tau_mu)
    s_m_mu = s_m_mu.transpose(1,0,2)
    a_m_mu, r_m_mu = a_m_mu.transpose(), r_m_mu.transpose()
    data_train = prepare_training_data(s0, s_m_mu, a_m_mu, r_m_mu)
    df = pd.DataFrame(data_train)
    df.to_csv('mdp_mountaincar.csv', header = False)
    print("damn")

    # s0_pi, s_m_pi, _, real_r_pi = my_traject.sample_trajectory(horizon, n_traj, tau_pi)
    
            
    # bandwidth = assian_bandwidth(horizon, reference)
    # sample_eps = assign_sample_eps(horizon, sample_reference)
    
    
    # dhatmu_ls = []
    # for t in range(horizon):
    #     kde = weighted_kde(bandwidth[t]).fit_d(s_m_mu[:,t])
    #     dhatmu_ls.append(kde)
        
    # dhatpi_ls = []
    # for t in range(horizon):
    #     kde = weighted_kde(bandwidth[t]).fit_d(s_m_pi[:,t])
    #     dhatpi_ls.append(kde)
    
    # qhatmu_ls = []
    # weight = compute_weight(my_traject.density, s0, a_m_mu[:,0],tau_pi, tau_mu)
    # try:
    #     kde = KernelDensity(kernel = "gaussian",bandwidth = bandwidth[0])
    # except:
    #     kde = KernelDensity(kernel = "gaussian",bandwidth = bandwidth[0])
    # kde.fit(np.hstack((s_m_mu[:,0],s0)),weight)
    # qhatmu_ls.append(kde)
    
    # for t in range(1, horizon):
    #     weight = compute_weight(my_traject.density, s_m_mu[:,t-1], a_m_mu[:,t],tau_pi,tau_mu)
    #     kde = KernelDensity(kernel = "gaussian", bandwidth=bandwidth[t])
    #     kde.fit(np.hstack((s_m_mu[:,t],s_m_mu[:,t-1])), weight)
    #     qhatmu_ls.append(kde)
                
            
    # if uniform_domain == True:
    #     representative,areas = create_representative_II(bandwidth, eps, s_m_pi, represent_nrow)
    # else:
    #     representative,areas = create_representative(bandwidth, num_eps, s_m_pi, represent_nrow)
    
    
    # samples = s_m_pi[:,0]
    # for t in range(horizon):
    #     print("begin generate step" + str(t))
    #     def conditional_pdf(x, y):
    #         return np.exp(qhatmu_ls[t+1].score_samples(np.hstack((x.reshape(-1,2), y.reshape(-1,2))))) / np.exp(dhatmu_ls[t].score_samples(y.reshape(-1,2)))
    #     my_cs= conditional_sampler(conditional_pdf, sample_eps[t],batch_size = batch_size)
    #     samples = my_cs.move_one_step(samples)
    #     try:
    #         samples = samples.squeeze(1)
    #     except:
    #         pass
    #     df = pd.DataFrame(samples)
    #     df.to_csv(after_sample_dir + str(t)+"policy pi.csv")
    #     plt.figure(dpi = 400)
    #     plt.scatter(samples[:,0], samples[:,1], label = "sampling result",s = 1)
    #     plt.scatter(s_m_pi[:,t + 1,0], s_m_pi[:,t + 1,1], label = "ground_truth", s = 1)
    #     plt.legend()
    #     plt.title("t = " + str(t))
    #     plt.show()
    #     plt.savefig(after_sample_dir + "\t = " + str(t) + ".png")
    #     plt.clf()
    
    # for t in range(horizon):
    #     print("begin generate step" + str(t))
    #     samples = dhatmu_ls[t].sample(n_traj)
    #     df = pd.DataFrame(samples)
    #     df.to_csv(after_sample_dir + str(t)+"policy mu.csv")
    #     plt.figure(dpi = 400)
    #     plt.scatter(samples[:,0], samples[:,1], label = "sampling result",s = 1)
    #     plt.scatter(s_m_pi[:,t + 1,0], s_m_pi[:,t + 1,1], label = "ground_truth", s = 1)
    #     plt.legend()
    #     plt.title("t = " + str(t))
    #     plt.show()
    #     plt.savefig(after_sample_dir + "\t = " + str(t) + ".png")
    #     plt.clf()
          
 

if __name__ == "__main__":
    main()



    
    
            
        
        
        
        
        
        
        
























