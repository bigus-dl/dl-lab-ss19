
#!/usr/bin/python3

from datetime import datetime
import numpy as np
import gym
import os
import json
import torch

from imitation_learning.agent.bc_agent import BCAgent
from imitation_learning.utils import *


def run_episode(env, agent, rendering=True, max_timesteps=1000, history=1):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events() 
    history_tensor = torch.zeros(1,history,96,96)

    while True:
        state = rgb2gray(state)
        state = torch.from_numpy(state)

        # shift if history is appended
        if history>1 :
            history_tensor[0,1:history-1] = history_tensor[0,0:history-2]
        
        # set channel 0 to current state
        history_tensor[0,0] = state
        a = agent.predict(history_tensor)
        print("network output, \tsoftmax : {} {}".format(a,torch.nn.functional.softmax(a,dim=0)))
        a = torch.nn.functional.softmax(a)
        
        a = torch.argmax(a).item()
        nstr =  "straight"  if a==0 else \
                "left"      if a==1 else \
                "right"     if a==2 else \
                "accel"     if a==3 else "brake"
                    
        # print("network output "+nstr)
        # run #1    = 0.8
        # pure run  = 1.0
        # 3 now8    = 0.8
        a = id_to_action(a, max_speed=0.7)
        next_state, r, done, info = env.step(a)   
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 15                  # number of episodes to test

    # TODO: load agent
    agent = BCAgent(history=1)
    mname = "h1_n"
    agent.load("./imitation_learning/snaps/snap"+mname)
    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering, history=1)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
    
    
    fname = "./imitation_learning/results/results_bc_agent_{}".format(mname)
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
