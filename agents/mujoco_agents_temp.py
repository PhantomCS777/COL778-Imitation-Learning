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
What is the criterion for my model

What is current train env steps?

Why arent we doing argmax anywhere


"""

def sample_trajectory_mod(
    env, exp_policy, new_policy, beta, max_length, render = False
):
    """Sample a rollout in the environment from a policy."""
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0

    while True:
        # render an image
        if render:
            # breakpoint()
            if hasattr(env, "sim"):
                img = env.sim.render(camera_name="track", height=500, width=500)[::-1]
            else:
                img = env.render(mode='single_rgb_array')
            image_obs.append(
                cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
            )

        #use the most recent ob to decide what to do
        in_ = torch.tensor(ob).to(ptu.device,torch.float)
        in_ = in_.unsqueeze(0)
        if(np.random.rand() < beta):
            ac = exp_policy.get_action(in_)
        else:
            ac = new_policy((normalize(in_, 0)))
            ac = ac.detach().cpu().numpy() # HINT: this is a numpy array
        ac = ac[0]

        # Take that action and get reward and next ob
        next_ob, rew, terminated, _ = env.step(ac)
        
        # Rollout can end due to done, or due to max_path_length
        steps += 1
        rollout_done = (terminated) or (steps >= max_length) # HINT: this is either 0 or 1

        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob  # jump to next timestep

        # end the rollout if the rollout ended
        if rollout_done:
            break

    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }



def sample_n_trajectories_mod(
    env, exp_policy, new_policy, beta, ntraj, max_length, render = False
):
    """Collect ntraj rollouts."""
    trajs = []
    for _ in range(ntraj):
        # collect rollout
        traj = sample_trajectory_mod(env, exp_policy, new_policy, beta, max_length, render)
        trajs.append(traj)
    return trajs




def exp_runner(exp_policy, trajectories):
    for traj in trajectories:
        for ind, obs in enumerate(traj["observation"]):
            traj["action"][ind] = exp_policy.get_action(torch.tensor(obs).to(ptu.device, torch.float))
            traj["action"][ind] = traj["action"][ind][0]
    return trajectories



def normalize(observation, observation_dim):
    """
    Normalize the observation to have mean 0 and std 1.
    """
    return observation
    observation = observation - observation.mean(dim=0)
    observation = observation / (observation.std(dim=0) + 1e-8)
    return observation



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
        self.replay_buffer = ReplayBuffer(10000) #you can set the max size of replay buffer if you want
        

        #initialize your model and optimizer and other variables you may need
        self.model = ptu.build_mlp(self.observation_dim, self.action_dim, self.hyperparameters["n_layers"], self.hyperparameters["hidden_size"],activation='relu')
        self.model.to(ptu.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr = 3e-4)
        self.criterion = nn.MSELoss()
        self.batch_size = self.hyperparameters["batch_size"]
        self.beta = 0.7
        

    def forward(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        normalize(observation, self.observation_dim)
        return self.model(observation)


    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        normalize(observation, self.observation_dim)    
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
        max_len = 10000
        num_traj = 200

        trajectories = sample_n_trajectories_mod(env, self.expert_policy, self.model, self.beta, num_traj, max_len)

        trajectories = exp_runner(self.expert_policy, trajectories)

        self.replay_buffer.add_rollouts(trajectories)
        batch = self.replay_buffer.sample_batch(self.batch_size)
        loss = self.update(batch["obs"], batch["acs"])
        self.beta = min(0.5,0.8*self.beta)

        return {'episode_loss': loss, 'trajectories': trajectories, 'current_train_envsteps': 0} #you can return more metadata if you want to


