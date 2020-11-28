from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from DRL.renderer import Renderer
import DRL.utils as utils
import torch
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
        self.decoder.eval()
        if args.cuda:
            self.decoder.cuda()
        for p in self.decoder.parameters():
            p.requires_grad = False
        # Content dataset
        content_dataset = datasets.ImageFolder(root=args.content_dataset_path,
                                               transform=transforms.Compose([
                                                   transforms.Resize(args.canvas_size),
                                                   transforms.CenterCrop(args.canvas_size),
                                                   transforms.ToTensor()]))
        content_trainset_size = int((1 - 1. / (1. + args.train_test_ratio)) * len(content_dataset))
        content_trainset = Subset(content_dataset, list(range(content_trainset_size)))
        content_testset = Subset(content_dataset, list(range(content_trainset_size, len(content_dataset))))
        self.content_trainloader = DataLoader(content_trainset, batch_size=self.batch_size,
                                              shuffle=True, num_workers=args.num_workers, drop_last=True)
        self.content_testloader = DataLoader(content_testset, batch_size=self.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
        self.content_trainiterator = iter(self.content_trainloader)
        # Style dataset
        style_dataset = datasets.ImageFolder(root=args.style_dataset_path,
                                             transform=transforms.Compose([
                                                 transforms.Resize(args.canvas_size),
                                                 transforms.CenterCrop(args.canvas_size),
                                                 transforms.ToTensor()]))
        self.style_dataloader = DataLoader(style_dataset, batch_size=self.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        self.style_dataiterator = iter(self.style_dataloader)

    def reset(self):
        # Content images
        self.content = next(self.content_trainiterator, 'terminal')
        if self.content == 'terminal':
            self.content_trainiterator = iter(self.content_trainloader)
            self.content = next(self.content_trainiterator, 'terminal')[0]
        else:
            self.content = self.content[0]
        # Style images
        self.style = next(self.style_dataiterator, 'terminal')
        if self.style == 'terminal':
            self.style_dataiterator = iter(self.style_dataloader)
            self.style = next(self.style_dataiterator, 'terminal')[0]
        else:
            self.style = self.style[0]
        # Canvas
        self.canvas = torch.zeros([self.batch_size, 3, self.canvas_size, self.canvas_size], dtype=torch.float32)
        if self.cuda:
            self.content = self.content.cuda()
            self.style = self.style.cuda()
            self.canvas = self.canvas.cuda()
        self.step = 0
        return self.observe()

    def observe(self):
        T = torch.ones([self.batch_size, 1, self.canvas_size, self.canvas_size], dtype=torch.float32) * self.step / self.episode_len
        if self.cuda:
            T = T.cuda()
        return torch.cat((self.canvas, self.content, self.style, T), dim=1)

    def step(self, action):
        with torch.no_grad():
            self.canvas = utils.decode(action, self.canvas, self.decoder, self.num_strokes, self.num_stroke_params)
        self.step += 1
        obs = self.observe()
        done = torch.as_tensor([self.step == self.episode_len] * self.batch_size, dtype=torch.float32)
        if self.cuda:
            done = done.cuda()
        return obs.detach(), done
