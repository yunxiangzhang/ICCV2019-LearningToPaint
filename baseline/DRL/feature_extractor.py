import torchvision.models as models
import torch.nn as nn
import torch


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, input):
        return (input - self.mean) / self.std


class FeatureExtractor:
    def __init__(self, args):
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        if args.cuda:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
        self.normalization = Normalization(self.mean, self.std)
        vgg19 = models.vgg19(pretrained=True).features.eval()
        self.model = nn.Sequential(self.normalization)
        self.load_model(vgg19)
        if args.cuda:
            self.model.cuda()
        for p in self.model.parameters():
            p.requires_grad = False

    def load_model(self, vgg19):
        i = 0
        for layer in vgg19.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            self.model.add_module(name, layer)
            if i >= 5:
                break
