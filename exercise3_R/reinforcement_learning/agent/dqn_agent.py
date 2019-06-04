import torch
import numpy as np
import math
from reinforcement_learning.agent.replay_buffer import ReplayBuffer
from imitation_learning.utils import *
import torchvision

class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, gamma=0.95, batch_size=64, epsilon=0.1, tau=0.01, lr=1e-4, burn_in=0, update="soft", buffer_cap=1e5,
                                    epsilon_upperbound=1, epsilon_lowerbound=0.1, epsilon_planning_episodes=100):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            gamma: discount factor of future rewards.
            batch_size: Number of samples per batch.
            tao: indicates the speed of adjustment of the slowly updated target network.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
            lr: learning rate of the optimizer
        """
        np.random.seed(None)
        self.device = torch.device("cpu")
        self.Q = Q.to(self.device)
        self.Q_target = Q_target.to(self.device)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q_target.eval()
        self.num_actions = num_actions

        # helps fill the replay buffer before reading batches
        self.burn_in = burn_in
        if burn_in > 0:
            print("burning in for {} samples".format(self.burn_in))
        self.history = 0
        self.replay_buffer = ReplayBuffer(capacity=buffer_cap)
        self.update = update
        
        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        # epsilon planning
        self.epsilon_uppperbound = epsilon_upperbound
        self.epsilon_lowerbound = epsilon_lowerbound
        self.epsilon_planning_episodes = epsilon_planning_episodes
        self.epsilon = self.epsilon_uppperbound
        print("performing cosine annealing from {} to {} for {} episodes".format(epsilon_upperbound,epsilon_lowerbound,epsilon_planning_episodes))

        self.loss_function = torch.nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)

    def soft_update(self, target, source, tau):
        '''
        soft gradient update for the target network
        '''        
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def epsilon_planner_update(self, episode) :
        '''
        epsilon planning using cosine annealing
        '''
        if episode >= self.epsilon_planning_episodes :
            self.epsilon = self.epsilon_lowerbound
            return
        # min = 0, max = 1
        factor = episode/self.epsilon_planning_episodes
        epsilon_temp = self.epsilon_uppperbound*math.cos(factor*math.pi/2)
        epsilon_temp = max(epsilon_temp,self.epsilon_lowerbound)
        self.epsilon = epsilon_temp

    def train(self, state, action, next_state, reward, terminal, episode):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """
        self.history+=1
        self.replay_buffer.add_transition(state=state, action=action, next_state=next_state, reward=reward, done=terminal)
        
        # burn in phase, replay buffer not full enough to sample from yet
        if self.history<= self.burn_in:
            if self.history==self.burn_in :
                print("burn-in phase finished")
            return
        
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = self.replay_buffer.next_batch(self.batch_size)
        batch_next_states = torch.from_numpy(batch_next_states).to(self.device)
        batch_next_states = batch_next_states.permute(0,3,1,2)
        batch_states = torch.from_numpy(batch_states).to(self.device)
        batch_states = batch_states.permute(0,3,1,2)

        batch_dones= torch.from_numpy(batch_dones).float().to(self.device)
        batch_rewards = torch.from_numpy(batch_rewards).float().to(self.device)
        batch_actions = torch.from_numpy(batch_actions).long().to(self.device)
        batch_actions = batch_actions.unsqueeze(dim=1)

        mat1 = torch.ones(batch_dones.shape)-batch_dones
        mat2 = torch.max(self.Q_target(batch_next_states), dim=1)[0]
        # mat2 = mat2.detach()
        td_targets = batch_rewards + self.gamma * mat1 * mat2
        td_targets = td_targets.detach()

        self.optimizer.zero_grad()
        self.Q.train()
        output = self.Q(batch_states)
        output = torch.gather(input=output, dim=1, index=batch_actions).squeeze()
        assert output.requires_grad

        loss = self.loss_function(td_targets, output)
        loss.backward()
        self.optimizer.step()
        
        # hard/soft update target net
        if self.update == "soft" :
            self.soft_update(self.Q_target, self.Q, self.tau)
        elif (self.history+1)%100 ==0:
            print("updating target net")
            self.Q_target.load_state_dict(self.Q.state_dict())

        # update epsilon
        self.epsilon_planner_update(episode=episode)
        # soft upgrade :
        #for w_policy, w_target in zip(self.Q.parameters(recurse=True), self.Q_target.parameters(recurse=True)) :
        #    w_target.data += (w_policy.data - w_target.data)*self.tau
        # self.soft_update(self.Q_target, self.Q, self.tau)

    @torch.no_grad()
    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)    
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id (0 = left, 1 = right) for cartpole, (0 = straight, 1=left, 2=right, 3=accelerate, 4=brake) for carracing
        """
        self.Q.eval()
        # if still in burn-in phase 
        if self.history<= self.burn_in:
            if np.random.uniform() > 0.5 :
                action_id=1
            else:
                action_id=0
            return action_id

        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            state = torch.from_numpy(state).to(self.device)
            state = state.float()
            output = self.Q(state)
            action_id = torch.argmax(output).item()
            
            return action_id
        else:
            if np.random.uniform() > 0.5 :
                action_id=1
            else:
                action_id=0
            return action_id

    @torch.no_grad()
    def act_racing(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)    
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id (0 = left, 1 = right) for cartpole, (0 = straight, 1=left, 2=right, 3=accelerate, 4=brake) for carracing
        """
        self.Q.eval()
        #            left            right           accel.          brake           straight
        #normal:     0.17568         0.08076         0.27278         0.02236         0.44842
        # if still in burn-in phase 

        if self.history<= self.burn_in:
            r = np.random.uniform()
            if   r<=0.3:
                action_id = ACCELERATE
            elif r<=0.71:
                action_id = STRAIGHT
            elif r<=0.88:
                action_id = LEFT
            elif r<=0.97:
                action_id = RIGHT
            else: 
                action_id = BRAKE
            return action_id

        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            state = torchvision.transforms.ToTensor()(state).to(self.device)
            state = state.unsqueeze(dim=0)
            output = self.Q(state)
            action_id = torch.argmax(output).item()
            return action_id
        else:
            r = np.random.uniform()
            if   r<=0.3:
                action_id = ACCELERATE
            elif r<=0.71:
                action_id = STRAIGHT
            elif r<=0.88:
                action_id = LEFT
            elif r<=0.97:
                action_id = RIGHT
            else: 
                action_id = BRAKE
            return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))

        '''
            if   r<=0.5:
                action_id = ACCELERATE
            elif r<=0.75:
                action_id = STRAIGHT
            elif r<=0.85:
                action_id = LEFT
            elif r<=0.95:
                action_id = RIGHT
            else: 
                action_id = BRAKE
            '''