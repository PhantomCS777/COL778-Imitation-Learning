import itertools
import torch
import random
from torch import nn
from torch import optim
import numpy as np
from tqdm import tqdm
import torch.distributions as distributions

from utils.replay_buffer import ReplayBuffer
import utils.utils as utils
from agents.base_agent import BaseAgent
import utils.pytorch_util as ptu
from policies.experts import load_expert_policy


# NOTE
"""
What is current train env steps?


"""



class ImitationAgent(BaseAgent):
    '''
    Please implement an Imitation Learning agent. Read train_agent.py to see how the class is used. 
    
    
    Note: 1) You may explore the files in utils to see what helper functions are available for you.
          2)You can add extra functions or modify existing functions. Dont modify the function signature of __init__ and train_iteration.  
          3) The hyperparameters dictionary contains all the parameters you have set for your agent. You can find the details of parameters in config.py.  
          4) You may use the util functions like utils/pytorch_util/build_mlp to construct your NN. You are also free to write a NN of your own. 
    
    Usage of Expert policy:
        Use self.expert_policy.get_action(observation:torch.Tensor) to get expert action for any given observation. 
        Expert policy expects a CPU tensors. If your input observations are in GPU, then 
        You can explore policies/experts.py to see how this function is implemented.
    '''

    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters ):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        self.replay_buffer = ReplayBuffer(5000) #you can set the max size of replay buffer if you want
        

        #initialize your model and optimizer and other variables you may need
        self.model = ptu.build_mlp(self.observation_dim, self.action_dim, self.hyperparameters["n_layers"], self.hyperparameters["hidden_size"])
        self.model.to(ptu.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr = 3e-4)
        self.criterion = nn.MSELoss()
        self.batch_size = self.hyperparameters["batch_size"]
        self.beta = 1
        

    def forward(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        return self.model(observation)


    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        return self.model(observation)
    
    
    def update(self, observations, actions):
        #*********YOUR CODE HERE******************
        self.model.train()
        observations = ptu.from_numpy(observations).float().to(ptu.device)
        actions = ptu.from_numpy(actions).to(ptu.device)

        loss = self.criterion(self.forward(observations), actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    


    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        if not hasattr(self, "expert_policy"):
            self.expert_policy, initial_expert_data = load_expert_policy(env, self.args.env_name)
            self.replay_buffer.add_rollouts(initial_expert_data)
            
            #to sample from replay buffer use self.replay_buffer.sample_batch(batch_size, required = <list of required keys>)
            # for example: sample = self.replay_buffer.sample_batch(32)
        
        #*********YOUR CODE HERE******************
        max_len = 10
        num_traj = 1

        if(np.random.rand() < self.beta):
            trajectories = utils.sample_n_trajectories(env, self.expert_policy, num_traj, max_len)
        else:
            trajectories = utils.sample_n_trajectories(env, self.model, num_traj, max_len)


        self.replay_buffer.add_rollouts(trajectories)
        batch = self.replay_buffer.sample_batch(self.batch_size)
        loss = self.update(batch["obs"], batch["acs"])


        return {'episode_loss': loss, 'trajectories': trajectories, 'current_train_envsteps': 0} #you can return more metadata if you want to


