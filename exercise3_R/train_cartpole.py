import sys
import os
import numpy as np
import gym
import itertools as it
from tensorboardX import SummaryWriter
import torch

from reinforcement_learning.agent.dqn_agent import DQNAgent
from reinforcement_learning.agent.networks import MLP
from imitation_learning.utils import EpisodeStats


def run_episode(env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    
    stats = EpisodeStats()        # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    while True:
        
        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:  
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)
        state = next_state
        
        if rendering:
            env.render()

        if terminal or step > max_timesteps: 
            break

        step += 1

    return stats

def train_online(env, agent, num_episodes, writer, model_dir="./models_cartpole", tensorboard_dir="./tensorboard", eval_cycle=20, num_eval_episodes=5):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("training agent")

    # training
    for i in range(num_episodes):
        print("episode: ",i)
        stats = run_episode(env, agent, deterministic=False, do_training=True)
        writer.add_scalar("left usage", stats.get_action_usage(0), i)
        writer.add_scalar("right usage", stats.get_action_usage(1), i)
        writer.add_scalar("episode_reward", stats.episode_reward, i)
        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent.pt"))
            eval_reward = 0
            for j in range(num_eval_episodes):
                eval_stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=False)
                eval_reward += eval_stats.episode_reward
            writer.add_scalar("mean_eval_reward", eval_reward/num_eval_episodes, i)


if __name__ == "__main__":

    eval_episodes = 5   # evaluate on 5 episodes
    eval_every = 10     # evaluate every 10 episodes
    tensorboard_dir="./reinforcement_learning/tensorboard"
    # hard or soft update
    name = "cartpole-hard-update"
    
    print("device : {}".format("cuda" if torch.cuda.is_available() else "cpu"))
    print("starting environment")
    env = gym.make("CartPole-v0").unwrapped

    state_dim = 4
    num_actions = 2
    
    print("starting tensorboard session")
    writer = SummaryWriter(os.path.join(tensorboard_dir,name))

    q_net      = MLP(state_dim, num_actions, hidden_dim=300)
    target_net = MLP(state_dim, num_actions, hidden_dim=300)
    # higher epsilon = more exploration
    # working 
    # agent = DQNAgent(q_net, target_net, num_actions=num_actions, gamma=0.95, batch_size=64, epsilon=0.05, tau=0.005, lr=1e-4, burn_in=256, update="hard", buffer_cap=5e3)
    # also working
    agent = DQNAgent(q_net, target_net, num_actions=num_actions, gamma=0.95, batch_size=64, epsilon=0.1, tau=0.01, lr=1e-4, burn_in=256, update="hard", buffer_cap=1e4)
    train_online(env, agent=agent, num_episodes=1000, writer=writer, model_dir="./reinforcement_learning/models_cartpole", eval_cycle=eval_every, num_eval_episodes=eval_episodes)
    writer.close()