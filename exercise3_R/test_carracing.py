import gym
from reinforcement_learning.agent.dqn_agent import DQNAgent
from reinforcement_learning.agent.networks import *
from datetime import datetime
from imitation_learning.utils import *
import json
import numpy as np
import os

np.random.seed(0)

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255

def run_episode(env, agent, deterministic, skip_frames=0,  do_training=True, rendering=False, max_timesteps=1000, history_length=0, episode_index=0):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    stats = EpisodeStats()
    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events() 

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1)
    while True:
        action_id = agent.act_racing(state=state, deterministic=deterministic)
        action = id_to_action(action_id, max_speed=0.5)
        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal: 
                 break
        
        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(96, 96, history_length + 1)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal, episode=episode_index)
        
        stats.step(reward, action_id)
        state = next_state
        if terminal or (step * (skip_frames + 1)) > max_timesteps: 
            break
        step += 1

    return stats

if __name__ == "__main__":

    env = gym.make("CarRacing-v0").unwrapped

    history_length =  3
    num_actions = 5
    model_dir = "./carracing"
    model_name = "dqn_carracing_eval_883"
    q_net      = CNN(inputs=history_length+1,outputs=num_actions)
    target_net = CNN(inputs=history_length+1,outputs=num_actions)
    # working 
    agent = DQNAgent(q_net, target_net, num_actions=num_actions, gamma=0.95, batch_size=16, tau=0.01, lr=5e-4, burn_in=0, update="soft", buffer_cap=1e4,
                                        epsilon_upperbound=1, epsilon_lowerbound=0.1, epsilon_planning_episodes=70)
    agent.load(os.path.join(model_dir, model_name))

    n_test_episodes = 15
    max_timesteps = 1000
    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, skip_frames=2, max_timesteps=max_timesteps, deterministic=True, do_training=False, rendering=True,history_length=history_length)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    if not os.path.exists("./results"):
        os.mkdir("./results")  

    fname = "./results/carracing_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

