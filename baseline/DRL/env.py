from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from renderer import Renderer
import torch
import utils
import os


class Paint:
    def __init__(self, args):
        self.cuda = args.cuda
        self.batch_size = args.batch_size
        self.episode_len = args.episode_len
        self.action_space_dim = args.action_space_dim
        self.obs_space_dim = (args.num_channels, args.canvas_size, args.canvas_size)
        self.canvas_size = args.canvas_size
        self.num_strokes = args.num_strokes
        self.num_stroke_params = args.num_stroke_params
        # Renderer
        self.decoder = Renderer()
        self.decoder.load_state_dict(torch.load(os.path.join(args.model_path, 'renderer.pkl')))
        if args.cuda:
            self.decoder.cuda()
        for p in self.decoder.parameters():
            p.requires_grad = False
        # Content and style datasets
        content_dataset = datasets.ImageFolder(root=args.content_dataset_path,
                                               transform=transforms.Compose([
                                                   transforms.Resize(args.canvas_size),
                                                   transforms.CenterCrop(args.canvas_size),
                                                   transforms.ToTensor()]))
        content_trainset_size = int((1 - 1. / (1. + args.train_test_ratio)) * len(content_dataset))
        content_trainset = Subset(content_dataset, list(range(content_trainset_size)))
        content_testset = Subset(content_dataset, list(range(content_trainset_size, len(content_dataset))))
        self.content_trainloader = DataLoader(content_trainset, batch_size=self.batch_size, shuffle=True, num_workers=args.num_workers)
        self.content_testloader = DataLoader(content_testset, batch_size=self.batch_size, shuffle=False, num_workers=args.num_workers)
        style_dataset = datasets.ImageFolder(root=args.style_dataset_path,
                                             transform=transforms.Compose([
                                                 transforms.Resize(args.canvas_size),
                                                 transforms.CenterCrop(args.canvas_size),
                                                 transforms.ToTensor()]))
        self.style_dataloader = DataLoader(style_dataset, batch_size=self.batch_size, shuffle=True, num_workers=args.num_workers)

    def reset(self):
        self.content = next(iter(self.content_trainloader))
        self.style = next(iter(self.style_dataloader))
        self.canvas = torch.zeros([self.batch_size, 3, self.canvas_size, self.canvas_size], dtype=torch.float32)
        if self.cuda:
            self.content.cuda()
            self.style.cuda()
            self.canvas.cuda()
        self.step = 0
        return self.observe()

    def observe(self):
        T = torch.ones([self.batch_size, 1, self.canvas_size, self.canvas_size], dtype=torch.float32) * self.step / self.episode_len
        if self.cuda:
            T.cuda()
        return torch.cat((self.canvas, self.content, self.style, T), dim=1)

    def step(self, action):
        self.canvas = utils.decode(action, self.canvas, self.decoder, self.num_strokes, self.num_stroke_params)
        self.step += 1
        obs = self.observe()
        done = torch.as_tensor([self.step == self.episode_len] * self.batch_size, dtype=torch.float32)
        if self.cuda:
            done.cuda()
        return obs.detach(), done
