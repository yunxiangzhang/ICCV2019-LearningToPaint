#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from DRL.ddpg import DDPG
from DRL.env import Paint
import numpy as np
import argparse
import random
import torch
import os

parser = argparse.ArgumentParser(description='Learning to Paint')
# Training-related parameters
parser.add_argument('--start-after', default=400, type=int, help='Number of steps to fill the replay buffer before training.')
parser.add_argument('--batch-size', default=96, type=int, help='Mini-batch size for training the agent.')
parser.add_argument('--num-workers', default=4, type=int, help='Number of parallel workers for loading data.')
parser.add_argument('--train-test-ratio', default=9.0, type=float, help='Ratio between training samples and test samples.')
parser.add_argument('--episode-len', default=40, type=int, help='Episode length.')
parser.add_argument('--eval-interval', default=50, type=int, help='Evaluation interval in terms of episodes.')
parser.add_argument('--num-eval-episodes', default=5, type=int, help='Number of episodes for evaluation.')
parser.add_argument('--num-env-steps', default=2000000, type=int, help='Number of agent-environment interaction steps.')
parser.add_argument('--num-train-steps', default=10, type=int, help='Number of training steps for each episode.')
parser.add_argument('--style-weight', default=1e6, type=float, help='Weight factor for the style loss.')
parser.add_argument('--content-weight', default=1.0, type=float, help='Weight factor for the content loss.')
# Model-related parameters
parser.add_argument('--discount', default=0.95**5, type=float, help='Discount factor.')
parser.add_argument('--num-channels', default=10, type=int, help='Number of channels for the state space.')
parser.add_argument('--replay-size', default=32000, type=int, help='Maximum length of replay buffer.')
parser.add_argument('--canvas-size', default=128, type=int, help='Canvas size.')
parser.add_argument('--polyak-factor', default=0.999, type=float, help='Interpolation factor in polyak averaging for target networks.')
parser.add_argument('--action-noise', default=0., type=float, help='Std of the Gaussian noise added to the policy during training.')
parser.add_argument('--num-strokes', default=5, type=int, help='Number of strokes in each action bundle.')
parser.add_argument('--num-stroke-params', default=13, type=int, help='Number of stroke parameters.')
# Other parameters
parser.add_argument('--path', default='./data', type=str, help='Path to all data, including trained models.')
parser.add_argument('--seed', default=1234, type=int, help='Random seed.')
args = parser.parse_args()


def train(agent, env, args):
    step = 0
    episode = 0
    episode_step = 0

    obs = None
    while step <= args.num_env_steps:
        # Update counters
        step += 1
        episode_step += 1
        # Reset the environment if necessary
        if obs is None:
            obs = env.reset()
            agent.reset(obs, args.action_noise)
        # The agent takes an action in the environment according to the current policy
        action = agent.select_action(obs, update=False, target=False)
        obs, done = env.step(action)
        agent.observe(obs, done)
        # End of episode
        print('Episode:', episode + 1)
        if episode_step == args.episode_len:
            if step > args.start_after:
                # Adjust learning rate
                if step < 10000 * args.episode_len:
                    lr = (3e-4, 1e-3)
                elif 10000 * args.episode_len <= step < 20000 * args.episode_len:
                    lr = (1e-4, 3e-4)
                else:
                    lr = (3e-5, 1e-4)
                # Train the agent
                for i in range(args.num_train_steps):
                    value, value_loss = agent.update_policy(lr)
                    print('Value: {:.2f} | Value loss: {:.2f}'.format(value, value_loss))
            obs = None
            episode_step = 0
            episode += 1


def main(args):
    # CUDA support
    args.cuda = torch.cuda.is_available()
    # Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    # Create folders if necessary
    args.model_path = os.path.join(args.path, 'models')
    args.content_dataset_path = os.path.join(args.path, 'dataset/content')
    args.style_dataset_path = os.path.join(args.path, 'dataset/style')
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.content_dataset_path):
        os.makedirs(args.content_dataset_path)
    if not os.path.exists(args.style_dataset_path):
        os.makedirs(args.style_dataset_path)
    # Create a painting environment and a painting agent
    args.action_space_dim = args.num_strokes * args.num_stroke_params
    env = Paint(args)
    agent = DDPG(args)
    train(agent, env, args)


if __name__ == "__main__":
    main(args)
