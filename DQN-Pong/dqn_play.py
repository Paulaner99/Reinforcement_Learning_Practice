from lib import wrappers
from lib import dqn_model

import argparse
import time

import numpy as np
import collections

import gym
import torch

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 30

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-s", "--env", default=DEFAULT_ENV_NAME, help="Envirnment name to use, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("-n", "--episodes", default=10, help="Number of episodes to run")
    parser.add_argument("-r", "--record", help="Directory for video")
    parser.add_argument("--no-vis", default=True, dest="vis", help="Disable visualization", action="store_false")
    args = parser.parse_args()
    
    env = wrappers.make_env(args.env)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    model_state = torch.load(args.model, map_location=lambda stg,_: stg)
    net.load_state_dict(model_state)

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    for _ in range(args.episodes):
        start_ts = time.time()
        if args.vis:
            env.render()

        state_v = torch.tensor(np.array([state]), copy=False)
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1

        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break
        if args.vis:
            delta = 1/FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)

        print("Total reward: %.2f" % total_reward)
        print("Action counts: ", c)
        if args.record:
            env.env.close()

