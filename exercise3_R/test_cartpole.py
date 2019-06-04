import os
from datetime import datetime
import gym
import json
from reinforcement_learning.agent.dqn_agent import DQNAgent
from train_cartpole import run_episode
from reinforcement_learning.agent.networks import *
import numpy as np

np.random.seed(0)

if __name__ == "__main__":
    
    model_dir="./reinforcement_learning/models_cartpole"
    model_name = "dqn_agent_best.pt"
    state_dim = 4
    num_actions = 2
    hidden = 300
    
    env = gym.make("CartPole-v0").unwrapped
    q_net      = MLP(4, num_actions, hidden_dim=hidden)
    target_net = MLP(4, num_actions, hidden_dim=hidden)
    agent = DQNAgent(q_net, target_net, num_actions=num_actions, batch_size=64, burn_in=128)
    agent.load(os.path.join(model_dir, model_name))
    # TODO: load DQN agent
    # ...
    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
    
    if not os.path.exists("./results"):
        os.mkdir("./results")  

    fname = "./results/cartpole_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

