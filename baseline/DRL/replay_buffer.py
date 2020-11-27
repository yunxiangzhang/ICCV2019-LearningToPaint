import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, args):
        self.obs_buf = torch.zeros((args.replay_size, args.num_channels, args.canvas_size, args.canvas_size), dtype=torch.float32)
        self.next_obs_buf = torch.zeros((args.replay_size, args.num_channels, args.canvas_size, args.canvas_size), dtype=torch.float32)
        self.act_buf = torch.zeros((args.replay_size, args.action_space_dim), dtype=torch.float32)
        self.done_buf = torch.zeros(args.replay_size, dtype=torch.float32)
        if args.cuda:
            self.obs_buf = self.obs_buf.cuda()
            self.next_obs_buf = self.next_obs_buf.cuda()
            self.act_buf = self.act_buf.cuda()
            self.done_buf = self.done_buf.cuda()
        self.ptr, self.size, self.max_size = 0, 0, args.replay_size
        self.batch_size = args.batch_size

    def store(self, obs, next_obs, act, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     next_obs=self.next_obs_buf[idxs],
                     act=self.act_buf[idxs],
                     done=self.done_buf[idxs])
        return batch
