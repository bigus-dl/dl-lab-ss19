
#!/usr/bin/python3

from datetime import datetime
import numpy as np
import gym
import os
import json
import torch
import torchvision.transforms

from imitation_learning.agent.bc_agent import BCAgent
from imitation_learning.utils import *


def run_episode(env, agent, rendering=True, max_timesteps=1000, history=1, max_speed=1):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events() 
    history_tensor = torch.zeros(1,history,96,96)

    while True:
        # preprocessing
        state = rgb2gray(state)
        state = state/255
        state = torch.from_numpy(state)
        
        # shift if history is appended
        if history>1 :
            for i in range(1,history) :
                history_tensor[0,history-i] = history_tensor[0,history-i-1]

        # set channel 0 to current state
        history_tensor[0,0] = state
        a = agent.predict(history_tensor)
        a = torch.nn.functional.softmax(a)
        a = torch.argmax(a).item()
        nstr =  "straight"  if a==0 else \
                "left"      if a==1 else \
                "right"     if a==2 else \
                "accel"     if a==3 else "brake"
                    
        # print("network output "+nstr)
        a = id_to_action(a, max_speed=max_speed)
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
    rendering = False                     
    
    # args for running episodes
    n_test_episodes = 15
    mxs = 0.7
    mname = "h5"
    hist = 5

    # TODO: load agent
    agent = BCAgent(history=hist)
    agent.load("./imitation_learning/snaps/snap"+mname)
    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering, history=hist, max_speed=mxs)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
    
    print ("mean : {}, std : {}".format(results["mean"],results["std"]))
    
    fname = "./imitation_learning/results/results_bc_agent_{}_{}".format(mname,mxs)
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
