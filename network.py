# -*- coding:utf-8 -*-
# @Time     : 2023/3/16  上午10:26
# @Author   : Rui Ma
# @Email    : ruima@std.uestc.edu.cn
# @File     : network.py
# @Software : PyCharm

import os
import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class ResNext101(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnext101_32x8d(pretrained=True)

        self.shared = nn.Sequential()
        for name, module in model.named_children():
            if name != 'fc':
                self.shared.add_module(name, module)

        self.shared.add_module(name='add_flatten', module=nn.Flatten())
        self.shared.add_module(name='add_linear', module=nn.Linear(in_features=2048, out_features=1))
        self.shared.add_module(name='add_sigmoid', module=nn.Sigmoid())

    def train_no_bn(self, mode=True):
        super(ResNext101, self).train(mode)
        for module in self.shared.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def forward(self, x):
        x = self.shared(x)
        return x


def init_dump(net_name):
    model = ResNext101()
    previous_masks = {}
    for module_idx, module in enumerate(model.shared.modules()):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            mask = torch.ByteTensor(module.weight.data.size()).fill_(0)
            mask = mask.cuda()
            previous_masks[module_idx] = mask

    checkpoint = {
        'previous_masks': previous_masks,
        'model': model
    }
    torch.save(checkpoint, './imagenet/{}.pt'.format(net_name))


if __name__ == '__main__':
    choice = 1
    if choice == 1:
        # 1
        if not os.path.exists('./imagenet/'):
            os.makedirs('./imagenet/')
        model_name = 'ResNext101'
        init_dump(model_name)
    elif choice == 2:
        # 2
        # net = ResNext101()
        net = models.resnext101_32x8d(pretrained=True)
        net.to('cuda')
        summary(net, (3, 256, 256))
        print(net)
    elif choice == 3:
        # 3
        # net = ResNext101()
        net = models.resnext101_32x8d(pretrained=True)
        net.to('cuda')
        i = 1
        for sub_name, sub_module in net.named_modules():
            print(str(i) + ': ', sub_name, sub_module)
            i += 1
    elif choice == 4:
        # 4
        load_name = './imagenet/ResNext101.pt'
        ckpt = torch.load(load_name)
        net = ckpt['model']
        net.to('cuda')
        summary(net, (3, 256, 256))
        print(net)
