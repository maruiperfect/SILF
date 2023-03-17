# -*- coding:utf-8 -*-
# @Time     : 2023/3/16  上午10:25
# @Author   : Rui Ma
# @Email    : ruima@std.uestc.edu.cn
# @File     : prune.py
# @Software : PyCharm

import os
import copy
import torch
import random
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class SparsePruner(object):
    def __init__(self, model, prune_once_perc, prune_twice_perc, previous_masks, train_bias=False, train_bn=False):
        self.model = model
        self.prune_once_perc = prune_once_perc
        self.prune_twice_perc = prune_twice_perc
        self.previous_masks = previous_masks
        self.train_bias = train_bias
        self.train_bn = train_bn

        valid_key = list(previous_masks.keys())[5]
        self.current_dataset_idx = previous_masks[valid_key][previous_masks[valid_key].le(100)].max()
        self.current_masks = None

    def pruning_mask_once(self, weights, previous_mask):
        """Ranks weights by magnitude. Set the value at the pruned position in the mask to 0. Returns pruned mask."""
        self.current_dataset_idx = self.current_dataset_idx.cuda()

        previous_mask = previous_mask.cuda()
        tensor = weights[previous_mask.eq(self.current_dataset_idx)]
        abs_tensor = tensor.abs()
        cutoff_rank = round(self.prune_once_perc * tensor.numel())

        if cutoff_rank != 0:
            cutoff_value = abs_tensor.view(-1).cpu()
            cutoff_value = cutoff_value.kthvalue(cutoff_rank)[0]
            cutoff_value = cutoff_value.cuda()

            remove_mask = weights.abs().le(cutoff_value) * previous_mask.eq(self.current_dataset_idx)

            if cutoff_value == 0:
                a = torch.nonzero(remove_mask == 1)

                index = []
                all_index = []
                for i in a:
                    for j in i:
                        index.append(j)
                    all_index.append(index)
                    index = []

                all_list = []
                for k in range(len(all_index)):
                    all_list.append(k)
                delete_list = random.sample(all_list, (len(all_index) - cutoff_rank))
                delete_list.sort()

                for p in range(len(all_index)):
                    if p in delete_list:
                        remove_mask[all_index[p]] = False

            previous_mask[remove_mask.eq(1)] = 0

        mask = previous_mask
        return mask

    def prune_once(self):
        """Return prune once masks."""
        print('Pruning once for dataset idx: %d' % self.current_dataset_idx)
        print('Pruning each layer by removing %.2f%% of values' % (100 * self.prune_once_perc))
        self.current_masks = {}
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = self.pruning_mask_once(module.weight.data, self.previous_masks[module_idx])
                self.current_masks[module_idx] = mask.cuda()
                # Set pruned weights to 0.
                weight = module.weight.data
                weight[self.current_masks[module_idx].eq(0)] = 0.0

        prune_masks = copy.deepcopy(self.current_masks)
        return prune_masks

    def pruning_mask_twice(self, weights, previous_mask):
        """
        Ranks the remaining weights by magnitude. Set the value at the pruned position in the mask to
        (100+current_dataset_idx). Returns pruned twice mask.
        """
        self.current_dataset_idx = self.current_dataset_idx.cuda()

        previous_mask = previous_mask.cuda()
        tensor = weights[previous_mask.eq(self.current_dataset_idx)]
        abs_tensor = tensor.abs()
        cutoff_rank = round(self.prune_twice_perc * tensor.numel())

        if cutoff_rank != 0:
            cutoff_value = abs_tensor.view(-1).cpu()
            cutoff_value = cutoff_value.kthvalue(cutoff_rank)[0]
            cutoff_value = cutoff_value.cuda()

            remove_mask = weights.abs().le(cutoff_value) * previous_mask.eq(self.current_dataset_idx)

            if cutoff_value == 0:
                a = torch.nonzero(remove_mask == 1)

                index = []
                all_index = []
                for i in a:
                    for j in i:
                        index.append(j)
                    all_index.append(index)
                    index = []

                all_list = []
                for k in range(len(all_index)):
                    all_list.append(k)
                delete_list = random.sample(all_list, (len(all_index) - cutoff_rank))
                delete_list.sort()

                for p in range(len(all_index)):
                    if p in delete_list:
                        remove_mask[all_index[p]] = False

            previous_mask[remove_mask.eq(1)] = 100 + self.current_dataset_idx

        mask = previous_mask
        return mask

    def prune_twice(self):
        """Return prune twice masks."""
        print('Pruning twice for dataset idx: %d' % self.current_dataset_idx)
        print('Pruning each layer by removing %.2f%% of values' % (100 * self.prune_twice_perc))
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = self.pruning_mask_twice(module.weight.data, self.previous_masks[module_idx])
                self.current_masks[module_idx] = mask.cuda()

        prune_twice_masks = copy.deepcopy(self.current_masks)
        return prune_twice_masks

    def make_grads_zero(self):
        """Set grads of fixed parameters to 0 (mask != current_dataset_idx)."""
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.current_masks[module_idx]
                if module.weight.grad is not None:
                    module.weight.grad.data[layer_mask.ne(self.current_dataset_idx)] = 0
                    if not self.train_bias:
                        if module.bias is not None:
                            module.bias.grad.data.fill_(0)
            elif 'BatchNorm' in str(type(module)):
                if not self.train_bn:
                    module.weight.grad.data.fill_(0)
                    module.bias.grad.data.fill_(0)

    def make_grads_zero_prune(self, masks):
        """Set grads of fixed parameters to 0 (mask != current_dataset_idx)."""
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = masks[module_idx]
                if module.weight.grad is not None:
                    module.weight.grad.data[layer_mask.ne(self.current_dataset_idx)] = 0
                    if not self.train_bias:
                        if module.bias is not None:
                            module.bias.grad.data.fill_(0)
            elif 'BatchNorm' in str(type(module)):
                if not self.train_bn:
                    module.weight.grad.data.fill_(0)
                    module.bias.grad.data.fill_(0)

    def make_grads_zero_prune_once(self, masks):
        """Set grads of fixed parameters to 0 (mask != (current_dataset_idx + 100))."""
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = masks[module_idx]
                if module.weight.grad is not None:
                    module.weight.grad.data[layer_mask.ne(self.current_dataset_idx + 100)] = 0
                    if not self.train_bias:
                        if module.bias is not None:
                            module.bias.grad.data.fill_(0)
            elif 'BatchNorm' in str(type(module)):
                if not self.train_bn:
                    module.weight.grad.data.fill_(0)
                    module.bias.grad.data.fill_(0)

    def make_pruned_zero(self):
        """Set partial weights to 0 (mask = 0)."""
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.current_masks[module_idx]
                module.weight.data[layer_mask.eq(0)] = 0.0

    def make_pruned_zero_last_task(self):
        """Set partial weights to 0 (mask = 0 and mask > 100)."""
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.current_masks[module_idx]
                module.weight.data[layer_mask.eq(0)] = 0.0
                module.weight.data[layer_mask.gt(100)] = 0.0

    def make_pruned_zero_prune(self, masks):
        """Set partial weights to 0 (mask = 0)."""
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = masks[module_idx]
                module.weight.data[layer_mask.eq(0)] = 0.0

    def make_pruned_zero_prune_twice(self, masks):
        """Set partial weights to 0 ((mask = 0) and (mask > 100))."""
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = masks[module_idx]
                module.weight.data[layer_mask.eq(0)] = 0.0
                module.weight.data[layer_mask.gt(100)] = 0.0

    def apply_mask(self):
        """Set partial weights to 0 (mask = 0)."""
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                mask = self.current_masks[module_idx].cuda()
                weight[mask.eq(0)] = 0.0

    def apply_mask_test(self, dataset_idx):
        """Set partial weights to 0 ((mask = 0) and (mask > dataset_idx))."""
        dataset_idx = dataset_idx.cuda()
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                mask = self.previous_masks[module_idx].cuda()
                weight[mask.eq(0)] = 0.0
                weight[mask.gt(dataset_idx)] = 0.0

    def apply_mask_test_prune_once(self, dataset_idx):
        """Set partial weights to 0 ((dataset_idx < mask < 100) and (mask > 100 + dataset_idx))."""
        dataset_idx = dataset_idx.cuda()
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                mask = self.previous_masks[module_idx].cuda()
                weight[mask.eq(0)] = 0.0
                a = mask.gt(dataset_idx)
                b = mask.lt(100)
                c = a * b
                weight[c] = 0.0
                d = mask.gt(100 + dataset_idx)
                weight[d] = 0.0

    def make_fine_tune_mask(self, mode):
        """Converts previously pruned weights into trainable weights for the current task."""
        self.current_dataset_idx += 1

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = self.previous_masks[module_idx]
                if mode == 'fine_tune':
                    mask[mask.eq(0)] = self.current_dataset_idx
                elif mode == 'last_task':
                    mask[mask.eq(100+(self.current_dataset_idx-3))] = self.current_dataset_idx

        self.current_masks = self.previous_masks

    @staticmethod
    def pruning_mask_relevance(weights, previous_mask, prune_ratio, dataset_idx):
        dataset_idx = torch.tensor(dataset_idx, dtype=torch.uint8)
        dataset_idx = dataset_idx.cuda()

        previous_mask = previous_mask.cuda()
        tensor = weights[previous_mask.eq(dataset_idx)]
        abs_tensor = tensor.abs()
        cutoff_rank = int(round(prune_ratio * tensor.numel()))

        if cutoff_rank != 0:
            cutoff_value = abs_tensor.view(-1).cpu()
            cutoff_value = cutoff_value.kthvalue(cutoff_rank)[0]
            cutoff_value = cutoff_value.cuda()

            remove_mask = weights.abs().le(cutoff_value) * previous_mask.eq(dataset_idx)

            if cutoff_value == 0:
                a = torch.nonzero(remove_mask == 1)

                index = []
                all_index = []
                for i in a:
                    for j in i:
                        index.append(j)
                    all_index.append(index)
                    index = []

                all_list = []
                for k in range(len(all_index)):
                    all_list.append(k)
                delete_list = random.sample(all_list, (len(all_index) - cutoff_rank))
                delete_list.sort()

                for p in range(len(all_index)):
                    if p in delete_list:
                        remove_mask[all_index[p]] = False

            previous_mask[remove_mask.eq(1)] = 0

    def prune_relevance(self, all_relevance_ratio):
        """Reuse partial parameters of previous tasks (reuse_ratio = 1 - prune_ratio)."""
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                for idx, prune_ratio in enumerate(all_relevance_ratio):
                    if prune_ratio != 0:
                        self.pruning_mask_relevance(module.weight.data, self.previous_masks[module_idx],
                                                    prune_ratio, idx + 1)
