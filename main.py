# -*- coding:utf-8 -*-
# @Time     : 2023/3/16  上午10:24
# @Author   : Rui Ma
# @Email    : ruima@std.uestc.edu.cn
# @File     : main.py
# @Software : PyCharm

import os
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from dataset import train_loader, test_loader
from prune import SparsePruner
from tensorboardX import SummaryWriter
from network import ResNext101

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(description='Scalable Incremental Learning Framework')

    # Basic settings
    parser.add_argument('--mode', type=str)
    parser.add_argument('--first_task', type=bool)
    parser.add_argument('--dataset_idx', type=int)
    parser.add_argument('--task_dataset', type=str)
    parser.add_argument('--fine_tune_epochs', type=int)
    parser.add_argument('--cycle_epochs', type=int)
    parser.add_argument('--prune_once_epochs', type=int)
    parser.add_argument('--prune_twice_epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_decay_every', type=int)
    parser.add_argument('--lr_decay_factor', type=float)
    parser.add_argument('--prune_once_perc', type=float)
    parser.add_argument('--prune_twice_perc', type=float)
    parser.add_argument('--train_data_loader')
    parser.add_argument('--test_data_loader')
    parser.add_argument('--model_load_path', type=str)
    parser.add_argument('--model_load_prefix', type=str)
    parser.add_argument('--model_save_path', type=str)
    parser.add_argument('--model_save_prefix', type=str)
    parser.add_argument('--log_save_path', type=str)
    parser.add_argument('--record_save_path', type=str)
    parser.add_argument('--complete_load_path', type=str)

    # Task relevance settings
    parser.add_argument('--relevance_save_path', type=str)
    parser.add_argument('--relevance_ratio', type=list, default=[])
    parser.add_argument('--all_relevance_ratio', type=list, default=[])
    parser.add_argument('--relevance_sr', type=list, default=[])
    parser.add_argument('--all_relevance_sr', type=list, default=[])
    parser.add_argument('--relevance_pl', type=list, default=[])
    parser.add_argument('--all_relevance_pl', type=list, default=[])
    parser.add_argument('--relevance_test_ratio', type=list, default=[])

    args = parser.parse_args()
    return args


class Manager(object):
    def __init__(self, model, previous_masks):
        self.model = model
        self.previous_masks = previous_masks
        self.train_data_loader = args.train_data_loader
        self.test_data_loader = args.test_data_loader
        self.criterion = nn.L1Loss()
        self.params_to_optimize = self.model.parameters()
        self.optimizer = optim.SGD(params=self.params_to_optimize, lr=args.lr)
        self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=args.lr_decay_every,
                                                   gamma=args.lr_decay_factor)
        self.pruner = SparsePruner(model=self.model, prune_once_perc=args.prune_once_perc,
                                   prune_twice_perc=args.prune_twice_perc, previous_masks=self.previous_masks)

    def train(self, fine_tune_epochs, train_mode):
        """Training on preset tasks."""
        print('-' * 50)
        print('Training Dataset: {}'.format(args.task_dataset))
        print('Training Mode: {}'.format(train_mode))
        print('-' * 50)

        print('Epoch\tTrain_Loss\tTrain_SRCC\tTrain_PLCC')

        self.model.train_no_bn()

        for fine_tune_epoch in range(fine_tune_epochs):
            fine_tune_epoch_loss, fine_tune_train_sr, fine_tune_train_pl = self.do_epoch()

            writer.add_scalar('Task---' + args.task_dataset[:-6] + '---Finetune', sum(fine_tune_epoch_loss) /
                              len(fine_tune_epoch_loss), fine_tune_epoch)

            print('%d\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (fine_tune_epoch + 1, sum(fine_tune_epoch_loss) / len(fine_tune_epoch_loss), fine_tune_train_sr,
                   fine_tune_train_pl))

            self.scheduler.step()
            self.save_model()

    def train_last_task(self, fine_tune_epochs, train_mode):
        """Training on additional tasks."""
        print('-' * 50)
        print('Training Dataset: {}'.format(args.task_dataset))
        print('Training Mode: {}'.format(train_mode))
        print('-' * 50)

        print('Epoch\tTrain_Loss\tTrain_SRCC\tTrain_PLCC')

        self.model.train_no_bn()

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = self.pruner.current_masks[module_idx].cuda()
                weight = module.weight.data
                weight[mask.gt(100)] = 0.0

        for fine_tune_epoch in range(fine_tune_epochs):
            fine_tune_epoch_loss, fine_tune_train_sr, fine_tune_train_pl = self.do_epoch_last_task()

            writer.add_scalar('Task---' + args.task_dataset[:-6] + '---LastTask', sum(fine_tune_epoch_loss) /
                              len(fine_tune_epoch_loss), fine_tune_epoch)

            print('%d\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (fine_tune_epoch + 1, sum(fine_tune_epoch_loss) / len(fine_tune_epoch_loss), fine_tune_train_sr,
                   fine_tune_train_pl))

            self.scheduler.step()
            self.save_model()

        # Save complete model
        before_relevance_ckpt = torch.load(args.model_load_path)
        before_relevance_masks = before_relevance_ckpt['previous_masks']
        before_relevance_model = before_relevance_ckpt['model'].cuda()
        before_relevance_model_list = list(before_relevance_model.shared.modules())

        after_relevance_ckpt = torch.load(args.model_save_prefix + '.pt')
        after_relevance_masks = after_relevance_ckpt['previous_masks']
        after_relevance_model = after_relevance_ckpt['model']
        after_relevance_model_list = list(after_relevance_model.shared.modules())

        task_idx = self.pruner.current_dataset_idx
        for module_idx, module in enumerate(after_relevance_model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                before_relevance_weight = before_relevance_model_list[module_idx].weight.data
                after_relevance_weight = after_relevance_model_list[module_idx].weight.data
                before_relevance_mask = before_relevance_masks[module_idx].cuda()
                after_relevance_mask = after_relevance_masks[module_idx].cuda()

                a = before_relevance_mask.ne(0)
                b = before_relevance_mask.lt(task_idx)
                c = a * b

                after_relevance_weight[c] = before_relevance_weight[c]
                after_relevance_mask[c] = before_relevance_mask[c]

        ckpt = {
            'previous_masks': after_relevance_masks,
            'model': after_relevance_model,
        }

        torch.save(ckpt, args.model_save_prefix + '_complete.pt')

    def prune_once_train(self, prune_once_epochs, prune_twice_masks, prune_once_model, train_mode):
        """Training prune once model."""
        print('-' * 50)
        print('Training Dataset: {}'.format(args.task_dataset))
        print('Training Mode: {}'.format(train_mode))
        print('-' * 50)

        print('Epoch\tTrain_Loss\tTrain_SRCC\tTrain_PLCC')

        self.model.train_no_bn()

        # Complete prune twice model
        prune_once_model_list = list(prune_once_model.shared.modules())
        prune_twice_model_checkpoint = torch.load(args.model_save_prefix + '_twice_model.pt')
        prune_twice_model = prune_twice_model_checkpoint['prune_twice_model']
        prune_twice_model_list = list(prune_twice_model.shared.modules())

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune_once_weight = prune_once_model_list[module_idx].weight.data
                prune_twice_weight = prune_twice_model_list[module_idx].weight.data
                prune_twice_mask = prune_twice_masks[module_idx].cuda()
                module.weight.data[prune_twice_mask.gt(100)] = prune_once_weight[
                    prune_twice_mask.gt(100)]
                module.weight.data[prune_twice_mask.lt(100)] = prune_twice_weight[
                    prune_twice_mask.lt(100)]

        # Training prune once model
        for prune_once_epoch in range(prune_once_epochs):
            prune_once_epoch_loss, prune_once_train_sr, prune_once_train_pl = self.do_epoch_prune_once(
                prune_twice_masks)

            print('%d\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (prune_once_epoch + 1, sum(prune_once_epoch_loss) / len(prune_once_epoch_loss),
                   prune_once_train_sr, prune_once_train_pl))

            self.scheduler.step()
            self.save_model()

    def prune_twice_train(self, prune_twice_epochs, prune_twice_masks, train_mode):
        """Training prune twice model."""
        print('-' * 50)
        print('Training Dataset: {}'.format(args.task_dataset))
        print('Training Mode: {}'.format(train_mode))
        print('-' * 50)

        print('Epoch\tTrain_Loss\tTrain_SRCC\tTrain_PLCC')

        self.model.train_no_bn()

        # Prune twice model
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = prune_twice_masks[module_idx].cuda()
                weight = module.weight.data
                weight[mask.gt(100)] = 0.0

        # Training prune twice model
        for prune_twice_epoch in range(prune_twice_epochs):
            prune_twice_epoch_loss, prune_twice_train_sr, prune_twice_train_pl = self.do_epoch_prune_twice(
                prune_twice_masks)

            print('%d\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (prune_twice_epoch + 1, sum(prune_twice_epoch_loss) / len(prune_twice_epoch_loss),
                   prune_twice_train_sr, prune_twice_train_pl))

            self.save_prune_twice_model()

    def save_complete_model(self):
        if not args.first_task:
            before_relevance_ckpt = torch.load(args.complete_load_path)
            before_relevance_masks = before_relevance_ckpt['previous_masks']
            before_relevance_model = before_relevance_ckpt['model']
            before_relevance_model_list = list(before_relevance_model.shared.modules())

            after_relevance_ckpt = torch.load(args.model_save_prefix + '.pt')
            after_relevance_masks = after_relevance_ckpt['previous_masks']
            after_relevance_model = after_relevance_ckpt['model']
            after_relevance_model_list = list(after_relevance_model.shared.modules())

            task_idx = self.pruner.current_dataset_idx
            for module_idx, module in enumerate(after_relevance_model.shared.modules()):
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    before_relevance_weight = before_relevance_model_list[module_idx].weight.data
                    after_relevance_weight = after_relevance_model_list[module_idx].weight.data
                    before_relevance_mask = before_relevance_masks[module_idx].cuda()
                    after_relevance_mask = after_relevance_masks[module_idx].cuda()

                    a = before_relevance_mask.ne(0)
                    b = before_relevance_mask.lt(task_idx)
                    c = a * b

                    after_relevance_weight[c] = before_relevance_weight[c]
                    after_relevance_mask[c] = before_relevance_mask[c]

            ckpt = {
                'previous_masks': after_relevance_masks,
                'model': after_relevance_model,
            }

            torch.save(ckpt, args.model_save_prefix + '_complete.pt')

    def do_epoch(self):
        epoch_loss = []
        p_scores = []
        t_scores = []
        for batch, label in self.train_data_loader:
            batch = batch.cuda()
            label = label.cuda()
            bs, n_crops, c, h, w = batch.size()

            self.model.zero_grad()
            output = self.model(batch.view(-1, c, h, w))
            output = output.view(bs, n_crops, -1).mean(1)
            output = output.view(-1)
            loss = self.criterion(output, label)
            batch_loss = loss.item()
            epoch_loss.append(batch_loss)
            loss.backward()

            p_scores = p_scores + output.cpu().tolist()
            t_scores = t_scores + label.cpu().tolist()

            self.pruner.make_grads_zero()
            self.optimizer.step()
            self.pruner.make_pruned_zero()

        train_sr, _ = stats.spearmanr(p_scores, t_scores)
        train_pl, _ = stats.pearsonr(p_scores, t_scores)
        return epoch_loss, train_sr, train_pl

    def do_epoch_last_task(self):
        epoch_loss = []
        p_scores = []
        t_scores = []
        for batch, label in self.train_data_loader:
            batch = batch.cuda()
            label = label.cuda()
            bs, n_crops, c, h, w = batch.size()

            self.model.zero_grad()
            output = self.model(batch.view(-1, c, h, w))
            output = output.view(bs, n_crops, -1).mean(1)
            output = output.view(-1)
            loss = self.criterion(output, label)
            batch_loss = loss.item()
            epoch_loss.append(batch_loss)
            loss.backward()

            p_scores = p_scores + output.cpu().tolist()
            t_scores = t_scores + label.cpu().tolist()

            self.pruner.make_grads_zero()
            self.optimizer.step()
            self.pruner.make_pruned_zero_last_task()

        train_sr, _ = stats.spearmanr(p_scores, t_scores)
        train_pl, _ = stats.pearsonr(p_scores, t_scores)
        return epoch_loss, train_sr, train_pl

    def do_epoch_prune_once(self, masks):
        epoch_loss = []
        p_scores = []
        t_scores = []
        for batch, label in self.train_data_loader:
            batch = batch.cuda()
            label = label.cuda()
            bs, n_crops, c, h, w = batch.size()

            self.model.zero_grad()
            output = self.model(batch.view(-1, c, h, w))
            output = output.view(bs, n_crops, -1).mean(1)
            output = output.view(-1)
            loss = self.criterion(output, label)
            batch_loss = loss.item()
            epoch_loss.append(batch_loss)
            loss.backward()

            p_scores = p_scores + output.cpu().tolist()
            t_scores = t_scores + label.cpu().tolist()

            self.pruner.make_grads_zero_prune_once(masks)
            self.optimizer.step()
            self.pruner.make_pruned_zero_prune(masks)

        train_sr, _ = stats.spearmanr(p_scores, t_scores)
        train_pl, _ = stats.pearsonr(p_scores, t_scores)
        return epoch_loss, train_sr, train_pl

    def do_epoch_prune_twice(self, masks):
        epoch_loss = []
        p_scores = []
        t_scores = []
        for batch, label in self.train_data_loader:
            batch = batch.cuda()
            label = label.cuda()
            bs, n_crops, c, h, w = batch.size()

            self.model.zero_grad()
            output = self.model(batch.view(-1, c, h, w))
            output = output.view(bs, n_crops, -1).mean(1)
            output = output.view(-1)
            loss = self.criterion(output, label)
            batch_loss = loss.item()
            epoch_loss.append(batch_loss)
            loss.backward()

            p_scores = p_scores + output.cpu().tolist()
            t_scores = t_scores + label.cpu().tolist()

            self.pruner.make_grads_zero_prune(masks)
            self.optimizer.step()
            self.pruner.make_pruned_zero_prune_twice(masks)

        train_sr, _ = stats.spearmanr(p_scores, t_scores)
        train_pl, _ = stats.pearsonr(p_scores, t_scores)
        return epoch_loss, train_sr, train_pl

    @torch.no_grad()
    def eval(self, dataset_idx, test_data_loader, model):
        p_scores = []
        t_scores = []

        if args.mode in ['fine_tune']:
            self.pruner.apply_mask()
        elif args.mode == 'test_prune_once':
            self.pruner.apply_mask_test_prune_once(dataset_idx)
        elif args.mode in ['last_task', 'test_prune_twice', 'test_last_task', 'relevance_ratio_cal']:
            self.pruner.apply_mask_test(dataset_idx)

        model.eval()
        for batch, label in test_data_loader:
            batch = batch.cuda()
            bs, n_crops, c, h, w = batch.size()
            output = model(batch.view(-1, c, h, w))
            output = output.view(bs, n_crops, -1).mean(1)
            output = output.view(-1)
            p_scores = p_scores + output.cpu().tolist()
            t_scores = t_scores + label.cpu().tolist()

        test_sr, _ = stats.spearmanr(p_scores, t_scores)
        test_pl, _ = stats.pearsonr(p_scores, t_scores)

        self.model.train_no_bn()

        return test_sr, test_pl

    def save_model(self):
        checkpoint = {
            'previous_masks': self.pruner.current_masks,
            'model': self.model,
        }

        torch.save(checkpoint, args.model_save_prefix + '.pt')

    def save_prune_mask_model(self, prune_once_masks, prune_twice_masks):
        prune_mask_model_checkpoint = {
            'prune_once_masks': prune_once_masks,
            'prune_twice_masks': prune_twice_masks,
            'model': self.model,
        }

        torch.save(prune_mask_model_checkpoint, args.model_save_prefix + '_mask_model.pt')

    def save_prune_twice_model(self):
        prune_twice_model_checkpoint = {
            'prune_twice_model': self.model,
        }

        torch.save(prune_twice_model_checkpoint, args.model_save_prefix + '_twice_model.pt')

    def prune(self):
        print('-' * 50)
        print('Begin Prune:')
        print('-' * 50)

        prune_once_masks = self.pruner.prune_once()
        prune_twice_masks = self.pruner.prune_twice()
        self.save_prune_mask_model(prune_once_masks, prune_twice_masks)

        for cycle_epoch in range(args.cycle_epochs):
            if cycle_epoch == 0:
                checkpoint = torch.load(args.model_save_prefix + '_mask_model.pt')
            else:
                checkpoint = torch.load(args.model_save_prefix + '.pt')

            prune_once_model = checkpoint['model']
            self.prune_twice_train(
                prune_twice_epochs=args.prune_twice_epochs, prune_twice_masks=prune_twice_masks,
                train_mode='prune twice')
            self.prune_once_train(
                prune_once_epochs=args.prune_once_epochs, prune_twice_masks=prune_twice_masks,
                prune_once_model=prune_once_model, train_mode='prune once')

        self.save_complete_model()

    def test(self):
        dataset_idx = args.dataset_idx
        dataset_idx = torch.tensor(dataset_idx, dtype=torch.uint8)
        if dataset_idx != 1:
            self.pruner.prune_relevance(args.relevance_test_ratio)
        test_sr, test_pl = self.eval(dataset_idx, self.test_data_loader, self.model)
        return test_sr, test_pl

    def relevance_ratio_cal(self):
        dataset_idx = args.dataset_idx
        dataset_idx = torch.tensor(dataset_idx, dtype=torch.uint8)
        if dataset_idx != 1:
            self.pruner.prune_relevance(args.relevance_test_ratio)

        test_sr, test_pl = self.eval(dataset_idx, self.train_data_loader, self.model)

        # The ratio here is defined as the pruning ratio, and the reuse ratio in formula (3)
        # is defined as (1 - ratio).
        if test_sr >= 0:
            ratio = 0
        else:
            ratio = -0.5 * test_sr

        test_sr = round(test_sr, 4)
        test_pl = round(test_pl, 4)
        ratio = round(ratio, 4)
        return ratio, test_sr, test_pl


def main():
    checkpoint = torch.load(args.model_load_path)
    previous_masks = checkpoint['previous_masks']
    model = checkpoint['model']
    model.cuda()
    manager = Manager(model, previous_masks)

    if args.mode == 'fine_tune':
        manager.pruner.make_fine_tune_mask(mode=args.mode)
        if not args.first_task:
            manager.pruner.prune_relevance(args.relevance_ratio)
        manager.train(fine_tune_epochs=args.fine_tune_epochs, train_mode='fine_tune')
    elif args.mode == 'prune':
        manager.prune()
    elif args.mode == 'last_task':
        manager.pruner.make_fine_tune_mask(mode=args.mode)
        manager.pruner.prune_relevance(args.relevance_ratio)
        manager.train_last_task(fine_tune_epochs=args.fine_tune_epochs, train_mode='last_task')
    elif args.mode in ['test_prune_once', 'test_prune_twice', 'test_last_task']:
        test_sr, test_pl = manager.test()
        return test_sr, test_pl
    elif args.mode == 'relevance_ratio_cal':
        ratio, test_sr, test_pl = manager.relevance_ratio_cal()
        return ratio, test_sr, test_pl


if __name__ == '__main__':
    args = parse_args()

    # Basic settings
    options = {
        'task_group': 'Group1',
        'batch_size': 5,
        'fine_tune_epochs': 40,
        'cycle_epochs': 10,
        'prune_once_epochs': 2,
        'prune_twice_epochs': 2,
        'prune_once_perc_list': [0.7, 0.5, 0],
        'prune_twice_perc_list': [0.4, 0.4, 0.4],
        'lr': 0.001,
        'lr_decay_every': 10,
        'lr_decay_factor': 0.5,
        'dataset_load_path': './Data/',
        'model_load_path': './imagenet/ResNext101.pt',
        'current_time': time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
    }

    # Datasets
    Datasets = []
    if options['task_group'] == 'Group1':
        Datasets = ['Challenge', 'CSIQ', 'LIEQ', 'KONIQ', 'LIVEMD_Merge', 'SHRQ_Aerial']
    elif options['task_group'] == 'Group2':
        Datasets = ['LIVE', 'SPAQ', 'KADID', 'IVIPC_DQA', 'TID2013', 'SHRQ_Regular']

    # Data path
    all_img_path = []
    all_mos_path = []
    for Dataset in Datasets:
        img_path = options['dataset_load_path'] + Dataset + '/'
        all_img_path.append(img_path)
        mos_path = img_path + Dataset + '.txt'
        all_mos_path.append(mos_path)

    # Dataloader
    all_train_data_loader = []
    all_test_data_loader = []
    for i in range(len(Datasets)):
        all_train_data_loader.append(
            train_loader(root=all_img_path[i], txt_root=all_mos_path[i], batch_size=options['batch_size'],
                         shuffle=True))
        all_test_data_loader.append(
            test_loader(root=all_img_path[i], txt_root=all_mos_path[i], batch_size=options['batch_size'],
                        shuffle=False))

    # Train, Prune, Test
    for j in range(len(Datasets)):
        # Model save path
        args.model_save_path = './checkpoints/' + options['task_group'] + '/' + options['current_time'] + '/' + \
                               Datasets[j] + '/'
        if not os.path.exists(args.model_save_path):
            os.makedirs(args.model_save_path)

        # Log save path
        args.log_save_path = './logs/' + options['task_group'] + '/' + options['current_time'] + '/' + \
                             Datasets[j] + '/'
        if not os.path.exists(args.log_save_path):
            os.makedirs(args.log_save_path)

        # Relevance save path
        args.relevance_save_path = './ratios/' + options['task_group'] + '/' + options['current_time'] + '/' + \
                                   Datasets[j] + '/'
        if not os.path.exists(args.relevance_save_path):
            os.makedirs(args.relevance_save_path)

        # Record save path
        args.record_save_path = './records/' + options['task_group'] + '/' + options['current_time'] + '/' + \
                                Datasets[j] + '/'
        if not os.path.exists(args.record_save_path):
            os.makedirs(args.record_save_path)

        writer = SummaryWriter(args.record_save_path)

        # Fine-tuned model save prefix
        fine_tuned_model_save_prefix = args.model_save_path + 'train'
        # Pruned model save prefix
        pruned_model_save_prefix = args.model_save_path + 'prune'

        # Model load path
        if j == 0:
            args.first_task = True
            args.model_load_path = options['model_load_path']
        elif j == 1:
            args.first_task = False
            args.model_load_path = args.model_load_prefix + '.pt'
        else:
            args.first_task = False
            args.model_load_path = args.model_load_prefix + '_complete.pt'

        # Public settings
        args.task_dataset = Datasets[j]
        args.train_data_loader = all_train_data_loader[j]
        args.test_data_loader = all_test_data_loader[j]
        args.fine_tune_epochs = options['fine_tune_epochs']
        args.cycle_epochs = options['cycle_epochs']
        args.prune_once_epochs = options['prune_once_epochs']
        args.prune_twice_epochs = options['prune_twice_epochs']
        args.lr = options['lr']
        args.lr_decay_every = options['lr_decay_every']
        args.lr_decay_factor = options['lr_decay_factor']

        # Relevance ratio calculation
        if j != 0:
            args.mode = 'relevance_ratio_cal'
            all_relevance_ratio = []
            all_relevance_sr = []
            all_relevance_pl = []
            for m in range(1, j + 1):
                args.dataset_idx = m
                args.relevance_test_ratio = args.all_relevance_ratio[m - 1]
                relevance_ratio, relevance_sr, relevance_pl = main()
                all_relevance_ratio.append(relevance_ratio)
                all_relevance_sr.append(relevance_sr)
                all_relevance_pl.append(relevance_pl)
            args.relevance_ratio = all_relevance_ratio
            args.relevance_sr = all_relevance_sr
            args.relevance_pl = all_relevance_pl
        args.all_relevance_ratio.append(args.relevance_ratio)
        args.all_relevance_sr.append(args.relevance_sr)
        args.all_relevance_pl.append(args.relevance_pl)
        with open(args.relevance_save_path + Datasets[j] + '_ratio' + '.txt', 'a') as file_object:
            file_object.write(Datasets[j] + '\n')
            file_object.write('Pruning Ratio:  ' + '\n')
            for item in args.all_relevance_ratio:
                file_object.write(str(item) + '\n')
        with open(args.relevance_save_path + Datasets[j] + '_sr' + '.txt', 'a') as file_object:
            file_object.write(Datasets[j] + '\n')
            file_object.write('Test SR:  ' + '\n')
            for item in args.all_relevance_sr:
                file_object.write(str(item) + '\n')
        with open(args.relevance_save_path + Datasets[j] + '_pl' + '.txt', 'a') as file_object:
            file_object.write(Datasets[j] + '\n')
            file_object.write('Test PL:  ' + '\n')
            for item in args.all_relevance_pl:
                file_object.write(str(item) + '\n')

        # Train
        if j >= 3:
            # Train on additional tasks
            args.mode = 'last_task'
            args.model_save_prefix = fine_tuned_model_save_prefix
            main()
            args.model_load_prefix = fine_tuned_model_save_prefix
        else:
            # Train on preset tasks
            args.mode = 'fine_tune'
            args.model_save_prefix = fine_tuned_model_save_prefix
            main()

            # Prune
            args.mode = 'prune'
            args.prune_once_perc = options['prune_once_perc_list'][j]
            args.prune_twice_perc = options['prune_twice_perc_list'][j]
            args.model_load_path = fine_tuned_model_save_prefix + '.pt'
            args.model_save_prefix = pruned_model_save_prefix
            main()
            args.model_load_prefix = pruned_model_save_prefix
            if args.first_task:
                args.complete_load_path = args.model_save_prefix + '.pt'
            else:
                args.complete_load_path = args.model_save_prefix + '_complete.pt'

        # Keep the test model and delete the rest of the intermediate models
        if j == 0:
            for file_name in os.listdir(args.model_save_path):
                if file_name != 'prune.pt':
                    os.remove(args.model_save_path + file_name)
        elif j >= 3:
            for file_name in os.listdir(args.model_save_path):
                if file_name != 'train_complete.pt':
                    os.remove(args.model_save_path + file_name)
        else:
            for file_name in os.listdir(args.model_save_path):
                if file_name != 'prune_complete.pt':
                    os.remove(args.model_save_path + file_name)

        # Test
        if j >= 3:
            # Test on additional tasks
            args.mode = 'test_last_task'
            args.model_load_path = fine_tuned_model_save_prefix + '_complete.pt'

            for n in range(j + 1):
                args.dataset_idx = n + 1
                args.test_data_loader = all_test_data_loader[n]
                args.relevance_test_ratio = args.all_relevance_ratio[n]
                Test_sr, Test_pl = main()
                Test_sr = format(Test_sr, '.4f')
                Test_pl = format(Test_pl, '.4f')
                detailed_log_name = args.log_save_path + args.mode + '/'
                if not os.path.exists(detailed_log_name):
                    os.makedirs(detailed_log_name)
                with open(detailed_log_name + Datasets[j] + '.txt', 'a') as file_object:
                    file_object.write(Datasets[n] + '\n')
                    file_object.write('sr:  ' + Test_sr + '\n')
                    file_object.write('pl:  ' + Test_pl + '\n')
        else:
            # Test on preset tasks
            args.mode = 'test_prune_once'
            if args.first_task:
                args.model_load_path = pruned_model_save_prefix + '.pt'
            else:
                args.model_load_path = pruned_model_save_prefix + '_complete.pt'

            for n in range(j + 1):
                args.dataset_idx = n + 1
                args.test_data_loader = all_test_data_loader[n]
                args.relevance_test_ratio = args.all_relevance_ratio[n]
                Test_sr, Test_pl = main()
                Test_sr = format(Test_sr, '.4f')
                Test_pl = format(Test_pl, '.4f')
                detailed_log_name = args.log_save_path + args.mode + '/'
                if not os.path.exists(detailed_log_name):
                    os.makedirs(detailed_log_name)
                with open(detailed_log_name + Datasets[j] + '.txt', 'a') as file_object:
                    file_object.write(Datasets[n] + '\n')
                    file_object.write('sr:  ' + Test_sr + '\n')
                    file_object.write('pl:  ' + Test_pl + '\n')

            args.mode = 'test_prune_twice'
            for n in range(j + 1):
                args.dataset_idx = n + 1
                args.test_data_loader = all_test_data_loader[n]
                args.relevance_test_ratio = args.all_relevance_ratio[n]
                Test_sr, Test_pl = main()
                Test_sr = format(Test_sr, '.4f')
                Test_pl = format(Test_pl, '.4f')
                detailed_log_name = args.log_save_path + args.mode + '/'
                if not os.path.exists(detailed_log_name):
                    os.makedirs(detailed_log_name)
                with open(detailed_log_name + Datasets[j] + '.txt', 'a') as file_object:
                    file_object.write(Datasets[n] + '\n')
                    file_object.write('sr:  ' + Test_sr + '\n')
                    file_object.write('pl:  ' + Test_pl + '\n')
