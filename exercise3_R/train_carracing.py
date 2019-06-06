# export DISPLAY=:0 
import os
import itertools as it
import numpy as np
import gym
from tensorboardX import SummaryWriter

from reinforcement_learning.agent.dqn_agent import DQNAgent
from reinforcement_learning.agent.networks import CNN
from imitation_learning.utils import *


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
            
            # forcing hands, taking names
            if step < 10 and action_id!=ACCELERATE :
                action_id = ACCELERATE
                action = id_to_action(action_id, max_speed=0.5)
                
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


def train_online(env, writer, agent, num_episodes=1000, history_length=0, 
                 model_dir="./reinforcement_learning/models_carracing", eval_cycle=20, num_eval_episodes=5, max_timesteps = 600, skip_frames=0) :
   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir) 

    time_steps = 350
    skip = skip_frames
    print("max_timesteps changed to ", time_steps)
    for i in range(num_episodes):
        print("epsiode %d" % i)
        
        # time step scheduling
        if i > 1 and i%50 ==0 :
            time_steps+=100
            if time_steps>max_timesteps :
                time_steps = max_timesteps
            print("max_timesteps changed to ", time_steps)
        # skip frame scheduling
        if i > 150 :
            skip = 4
        elif i > 200 :
            skip = 3
        elif i > 250 :
            skip = 2
        elif i > 300 :
            skip = 1
        elif i > 400 :
            skip = 0

        stats = run_episode(env, agent, max_timesteps=time_steps, skip_frames=skip, rendering=False, deterministic=False, 
                            do_training=True, history_length=history_length, episode_index=i)

        writer.add_scalar("straight", stats.get_action_usage(STRAIGHT), i)
        writer.add_scalar("left", stats.get_action_usage(LEFT), i)
        writer.add_scalar("right", stats.get_action_usage(RIGHT), i)
        writer.add_scalar("accel", stats.get_action_usage(ACCELERATE), i)
        writer.add_scalar("brake", stats.get_action_usage(BRAKE), i)
        writer.add_scalar("episode_reward", stats.episode_reward, i)

        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_carracing_agent.pt"))
            eval_reward = 0
            for j in range(num_eval_episodes):
                eval_stats = run_episode(env, agent, skip_frames=skip, max_timesteps=max_timesteps, deterministic=True, do_training=False,
                                                     rendering=(j==0), history_length=history_length)
                eval_reward += eval_stats.episode_reward
            writer.add_scalar("mean_eval_reward", eval_reward/num_eval_episodes, i)
            if eval_reward/num_eval_episodes > 700 :
                agent.save(os.path.join(model_dir, "dqn_carracing_eval_"+str(int(eval_reward/num_eval_episodes))))



def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255

if __name__ == "__main__":

    print("starting environment")
    env = gym.make('CarRacing-v0').unwrapped
    
    tensorboard_dir="./reinforcement_learning/tensorboard"
    model_dir = "./carracing"
    name = "ddqn-softh4-b16-force10-skip-2"
    num_eval_episodes = 5
    eval_cycle = 10
    history = 3
    num_actions = 5
    
    print("starting tensorboard session")
    writer = SummaryWriter(os.path.join(tensorboard_dir,name))
    q_net      = CNN(inputs=history+1,outputs=num_actions)
    target_net = CNN(inputs=history+1,outputs=num_actions)
    # working 
    agent = DQNAgent(q_net, target_net, num_actions=num_actions, gamma=0.95, batch_size=16, tau=0.01, lr=5e-4, burn_in=16, update="soft", buffer_cap=1e4,
                                        epsilon_upperbound=1, epsilon_lowerbound=0.1, epsilon_planning_episodes=70)
    
    train_online(env, agent=agent, num_episodes=1000, writer=writer, history_length=history, model_dir=model_dir, 
                      eval_cycle=eval_cycle, num_eval_episodes=num_eval_episodes, max_timesteps=1000, skip_frames=5)
    writer.close()  

