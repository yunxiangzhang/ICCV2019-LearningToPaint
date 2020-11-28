from DRL.feature_extractor import FeatureExtractor
from DRL.replay_buffer import ReplayBuffer
from DRL.renderer import Renderer
from DRL.critic import Critic
from DRL.actor import Actor

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import DRL.utils as utils
import torch
import os


class DDPG:
    def __init__(self, args):
        self.model_path = args.model_path
        self.canvas_size = args.canvas_size
        self.num_strokes = args.num_strokes
        self.num_stroke_params = args.num_stroke_params
        self.style_weight = args.style_weight
        self.content_weight = args.content_weight
        self.episode_len = args.episode_len
        self.batch_size = args.batch_size
        self.polyak_factor = args.polyak_factor
        self.discount = args.discount
        self.cuda = args.cuda
        # Actor and its target
        self.actor = Actor(12, 18, args.action_space_dim)  # 3(content)+3(style)+3(canvas)+1(step)+2(coordconv)
        self.actor_target = Actor(12, 18, args.action_space_dim)
        self.actor_target.eval()
        for p in self.actor_target.parameters():
            p.requires_grad = False
        utils.hard_update(self.actor_target, self.actor)
        # Critic and its target
        self.critic = Critic(12 + 3, 18, 1)  # Add the second most recent canvas for better prediction
        self.critic_target = Critic(12 + 3, 18, 1)
        self.critic_target.eval()
        for p in self.critic_target.parameters():
            p.requires_grad = False
        utils.hard_update(self.critic_target, self.critic)
        # Renderer
        self.decoder = Renderer()
        self.decoder.load_state_dict(torch.load(os.path.join(args.model_path, 'renderer.pkl')))
        self.decoder.eval()
        for p in self.decoder.parameters():
            p.requires_grad = False
        # Feature extractor
        self.feature = FeatureExtractor(args)
        # CoordConv
        coord = torch.zeros([1, 2, self.canvas_size, self.canvas_size])
        for i in range(self.canvas_size):
            for j in range(self.canvas_size):
                coord[0, 0, i, j] = float(i) / (self.canvas_size - 1)
                coord[0, 1, i, j] = float(j) / (self.canvas_size - 1)
        self.coord = coord
        # Select the correct device
        if self.cuda:
            self.cuda()
        # Replay buffer
        self.memory = ReplayBuffer(args)
        # Most recent state and action
        self.state = None
        self.action = None
        # Optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-2)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-2)

    def observe(self, obs, done):
        for i in range(self.batch_size):
            self.memory.store(self.state[i], obs[i], self.action[i], done[i])
        self.state = obs

    def select_action(self, state, update=False, target=False):
        state = torch.cat((state, self.coord.expand(state.shape[0], 2, self.canvas_size, self.canvas_size)), dim=1)
        if update:
            action = self.actor(state)
        else:
            with torch.no_grad():
                if target:
                    action = self.actor_target(state)
                else:
                    self.eval()
                    action = self.actor(state)
                    self.train()
                    self.action = action
        return action

    def compute_style_transfer_loss(self, canvas, content, style):
        content_features = []
        style_features = []

        def content_hook(self, inputs, outputs):
            content_features.append(outputs)

        def style_hook(self, inputs, outputs):
            style_features.append(outputs)

        def add_hooks(feature_extractor):
            handles = []
            i = 0
            for layer in feature_extractor.model.children():
                if isinstance(layer, nn.Conv2d):
                    i += 1
                    name = 'conv_{}'.format(i)
                    if name in feature_extractor.content_layers:
                        handles.append(layer.register_forward_hook(content_hook))
                    if name in feature_extractor.style_layers:
                        handles.append(layer.register_forward_hook(style_hook))
            return handles

        handles = add_hooks(self.feature)
        self.feature.model(canvas)
        self.feature.model(style)
        self.feature.model(content)
        for handle in handles:
            handle.remove()

        content_loss = F.mse_loss(content_features[0], content_features[2].detach())
        style_loss = F.mse_loss(utils.gramm_matrix(style_features[0]), utils.gramm_matrix(style_features[5]).detach())
        for i in range(1, 5):
            style_loss += F.mse_loss(utils.gramm_matrix(style_features[i]), utils.gramm_matrix(style_features[i + 5]).detach())
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss
        return total_loss

    def evaluate(self, state, action, target=False):
        # Current state
        canvas_before = state[:, 0:3]
        content = state[:, 3:6]
        style = state[:, 6:9]
        T = state[:, 9:10]
        # Canvas status after taking actions
        canvas_after = utils.decode(action, canvas_before, self.decoder, self.num_strokes, self.num_stroke_params)
        # Style transfer loss
        loss_before = self.compute_style_transfer_loss(canvas_before, content, style)
        loss_after = self.compute_style_transfer_loss(canvas_after, content, style)
        # Reward
        reward = loss_before - loss_after
        # Construct merged state
        coord = self.coord.expand(state.shape[0], 2, self.canvas_size, self.canvas_size)
        merged_state = torch.cat([canvas_before, canvas_after, content, style, T + (1.0 / self.episode_len), coord], dim=1)
        if target:
            value = self.critic_target(merged_state)
        else:
            value = self.critic(merged_state)
        return value + reward, reward

    def update_policy(self, lr):
        # Update learning rates
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = lr[0]
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr[1]
        # Randomly sample a batch of training data from the replay buffer
        state, next_state, action, done = self.memory.sample_batch()
        # Compute target value
        with torch.no_grad():
            next_action = self.select_action(next_state, update=False, target=True)
            target_value, _ = self.evaluate(next_state, next_action, target=True)
            target_value = self.discount * ((1 - done).view(-1, 1)) * target_value
        # Compute current value
        current_value, reward = self.evaluate(state, action, target=False)
        target_value += reward.detach()
        # Value update
        self.critic_optim.zero_grad()
        criterion = nn.MSELoss()
        value_loss = criterion(current_value, target_value)
        value_loss.backward(retain_graph=True)
        self.critic_optim.step()
        # Policy update
        self.actor_optim.zero_grad()
        action = self.select_action(state, update=True, target=False)
        value, _ = self.evaluate(state.detach(), action, target=False)
        policy_loss = -value.mean()
        policy_loss.backward(retain_graph=True)
        self.actor_optim.step()
        # Update target networks
        utils.soft_update(self.actor_target, self.actor, self.polyak_factor)
        utils.soft_update(self.critic_target, self.critic, self.polyak_factor)
        return -policy_loss, value_loss

    def reset(self, obs, action_noise):
        self.state = obs
        self.action_noise = action_noise

    def save_model(self):
        self.actor.cpu()
        self.critic.cpu()
        torch.save(self.actor.state_dict(), os.path.join(self.model_path, 'actor.pkl'))
        torch.save(self.critic.state_dict(), os.path.join(self.model_path, 'critic.pkl'))
        if self.cuda:
            self.actor.cuda()
            self.critic.cuda()

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def train(self):
        self.actor.train()
        self.critic.train()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()
        self.decoder.cuda()
        self.coord = self.coord.cuda()
