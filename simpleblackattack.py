import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import utils
import math
import random
import numpy as np
import argparse
import os

class Simba:
    def __init__(self, model, max_iters=300, freq_dims=32, stride=7, epsilon=0.05, linf_bound=0.0, order='rand', targeted=True, pixel_attack=False, log_every=40):
        self.model =model
        self.max_iters = max_iters
        self.freq_dims = freq_dims
        self.stride = stride
        self.epsilon = epsilon
        self.linf_bound = linf_bound
        self.order = order
        self.targeted = targeted
        self.pixel_attack = pixel_attack
        self.log_every = log_every
        self.image_size = 32
        self.dataset = 'cifar'
        self.device = next(model.parameters()).device
        self.model.eval()
    
    def expand_vector(self, x, size):
        batch_size = x.size(0)
        x = x.view(-1, 3, size, size)
        z = torch.zeros(batch_size, 3, self.image_size, self.image_size)
        z[:, :, :size, :size] = x
        return z
        
    def normalize(self, x):
        return utils.apply_normalization(x, self.dataset)

    def get_probs(self, x, y):
        output = self.model(self.normalize(x.cuda()))

        probs = torch.index_select(F.softmax(output, dim=-1).data, 1, y)
        return torch.diag(probs)
    
    def get_preds(self, x):
        x = self.normalize(x.cuda())
        # print(f"[DEBUG] x.min: {x.min()}, images.max: {x.max()}")
        output = self.model(x.cuda())
        # output = self.model(self.normalize(x.cuda())).cpu()
        _, preds = output.data.max(1)
        return preds
    # def get_probs(self, x, y):
    #     output = self.model(x.cuda())
    #     probs = torch.index_select(F.softmax(output, dim=-1).data, 1, y)
    #     return torch.diag(probs)
    
    # def get_preds(self, x):
    #     output = self.model(x.cuda())
    #     _, preds = output.data.max(1)
    #     return preds

    def simba_batch(self, images_batch, labels_batch, max_iters, freq_dims, stride, epsilon, linf_bound, order='rand', targeted=True, pixel_attack=False, log_every=40):
        batch_size = images_batch.size(0)
        image_size = images_batch.size(2)
        images_batch = images_batch.cuda()
        labels_batch = labels_batch.cuda()
        # print(type(images_batch))
        # print(f"[DEBUG] image_batch.min: {images_batch.min()}, image_batch.max: {images_batch.max()}")
        assert self.image_size == image_size
        # sample a random ordering for coordinates independently per batch element
        if order == 'rand':
            indices = torch.randperm(3 * freq_dims * freq_dims)[:max_iters]
        elif order == 'diag':
            indices = utils.diagonal_order(image_size, 3)[:max_iters]
        elif order == 'strided':
            indices = utils.block_order(image_size, 3, initial_size=freq_dims, stride=stride)[:max_iters]
        else:
            indices = utils.block_order(image_size, 3)[:max_iters]
        if order == 'rand':
            expand_dims = freq_dims
        else:
            expand_dims = image_size
        n_dims = 3 * expand_dims * expand_dims
        x = torch.zeros(batch_size, n_dims)
        # logging tensors
        probs = torch.zeros(batch_size, max_iters)
        succs = torch.zeros(batch_size, max_iters)
        # queries = torch.zeros(batch_size)
        iteration_nums = torch.ones(batch_size).long().to(self.device)
        queries = torch.zeros(batch_size, max_iters).to(self.device)
        prev_probs = self.get_probs(images_batch, labels_batch)
        preds = self.get_preds(images_batch)
        samples = preds.eq(labels_batch)
        correct_samples = samples.sum()  # 计算正确的样本数量
        print(f"Number of correct samples: {correct_samples.item()}")
        if pixel_attack:
            trans = lambda z: z
        else:
            trans = lambda z: utils.block_idct(z, block_size=image_size, linf_bound=linf_bound)
        remaining_indices = torch.arange(0, batch_size).long().to(self.device)
        for k in range(max_iters):
            dim = indices[k]
            # print("remaining device:", remaining_indices.device)
            # print("images device:", images_batch.device)
            # print("labels_batch device:", labels_batch.device)
            
            # print("expand_dims device:", expand_dims.device)
            x = x.to(self.device)
            # print("x device:", x.device)
            expanded = (images_batch[remaining_indices] + trans(self.expand_vector(x[remaining_indices], expand_dims).to(self.device)).to(self.device)).clamp(0, 1)
            perturbation = trans(self.expand_vector(x, expand_dims))
            preds_next = self.get_preds(expanded)
            preds[remaining_indices] = preds_next
            # print("remaining device:", remaining.device)
            # print("preds device:", preds.device)
            # print("labels_batch device:", labels_batch.device)
            if targeted:
                remaining = preds.ne(labels_batch)
            else:
                remaining = preds.eq(labels_batch)
            # check if all images are misclassified and stop early
            if remaining.sum() == 0:
                adv = (images_batch + trans(self.expand_vector(x, expand_dims))).clamp(0, 1)
                probs_k = self.get_probs(adv, labels_batch)
                probs[:, k:] = probs_k.unsqueeze(1).repeat(1, max_iters - k)
                succs[:, k:] = torch.ones(batch_size, max_iters - k)
                queries[:, k:] = torch.zeros(batch_size, max_iters - k)
                break
            # print("remaining device:", remaining.device)
            # print("remaining_indices device:", remaining_indices.device)
            remaining_indices = torch.arange(0, batch_size, device = self.device)[remaining].long()
            if k > 0:
                succs[:, k] = ~remaining
            diff = torch.zeros(remaining.sum(), n_dims)
            diff[:, dim] = epsilon
            diff = diff.to(self.device)
            left_vec = x[remaining_indices] - diff
            right_vec = x[remaining_indices] + diff
            # trying negative direction
            adv = (images_batch[remaining_indices] + trans(self.expand_vector(left_vec, expand_dims).to(self.device)).to(self.device)).clamp(0, 1)
            left_probs = self.get_probs(adv, labels_batch[remaining_indices])
            queries_k = torch.zeros(batch_size).to(self.device)
            # increase query count for all images
            queries_k[remaining_indices] += 1
            prev_probs = prev_probs.to(self.device)
            # print("prev_probs device:", prev_probs.device)
            # print("remaining_indices device:", remaining_indices.device)
            # print("left_probs device:", left_probs.device)
            if targeted:
                improved = left_probs.gt(prev_probs[remaining_indices])
            else:
                improved = left_probs.lt(prev_probs[remaining_indices])
            # only increase query count further by 1 for images that did not improve in adversarial loss没有改进的样本计数+1
            if improved.sum() < remaining_indices.size(0):
                queries_k[remaining_indices[~improved]] += 1
            # try positive directions
            adv = (images_batch[remaining_indices] + trans(self.expand_vector(right_vec, expand_dims).to(self.device)).to(self.device)).clamp(0, 1)
            right_probs = self.get_probs(adv, labels_batch[remaining_indices])
            if targeted:
                right_improved = right_probs.gt(torch.max(prev_probs[remaining_indices], left_probs))
            else:
                right_improved = right_probs.lt(torch.min(prev_probs[remaining_indices], left_probs))
            probs_k = prev_probs.clone()
            # update x depending on which direction improved
            if improved.sum() > 0:
                left_indices = remaining_indices[improved]
                left_mask_remaining = improved.unsqueeze(1).repeat(1, n_dims)
                x[left_indices] = left_vec[left_mask_remaining].view(-1, n_dims)
                probs_k[left_indices] = left_probs[improved]
            if right_improved.sum() > 0:
                right_indices = remaining_indices[right_improved]
                right_mask_remaining = right_improved.unsqueeze(1).repeat(1, n_dims)
                x[right_indices] = right_vec[right_mask_remaining].view(-1, n_dims)
                probs_k[right_indices] = right_probs[right_improved]
            probs[:, k] = probs_k.to(self.device)
            queries[:, k] = queries_k.to(self.device)
            # queries += queries_k
            prev_probs = probs[:, k].to(self.device)
            iteration_nums[remaining_indices] += 1
            if (k + 1) % log_every == 0 or k == max_iters - 1:
                print('Iteration %d: queries = %.4f, prob = %.4f, remaining = %.4f' % (
                        k + 1, queries.sum(0).mean(), probs[:, k].mean(), remaining.float().mean()))
        expanded = (images_batch + trans(self.expand_vector(x, expand_dims).to(self.device)).to(self.device)).clamp(0, 1)
        preds = self.get_preds(expanded)
        if targeted:
            remaining = preds.ne(labels_batch)
        else:
            remaining = preds.eq(labels_batch)
        succs[:, max_iters-1] = ~remaining
        return expanded, probs, succs, queries,iteration_nums



def simpleblackbox(model, images, labels, max_iters=300, freq_dims=32, stride=7, epsilon=0.05, linf_bound=0.0, order='rand', targeted=False, pixel_attack=False, log_every=40):

    simpleBlackBox_Attack = Simba(
        model=model,
        max_iters=max_iters,
        freq_dims=freq_dims,
        stride=stride,
        epsilon=epsilon,
        linf_bound=linf_bound,
        order=order,
        targeted=targeted,
        pixel_attack=pixel_attack,
        log_every=log_every
    )
    images_batch = images
    labels_batch = labels
    _, _, _, _, iteration_nums = simpleBlackBox_Attack.simba_batch(images_batch, labels_batch, max_iters=300, freq_dims=32, stride=7, epsilon=0.05, linf_bound=0.0, order='rand', targeted=False, pixel_attack=False, log_every=40)
    return iteration_nums