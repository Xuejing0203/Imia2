import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import utils
import math
import random
import numpy as np
import argparse
import os


class PGDAttack:
    def __init__(self, model, eps=3/255,
                 alpha=0.001, steps=50, random_start=True, targeted=False):
        self.model =model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.targeted = targeted
        self.device = next(model.parameters()).device
        self.loss_fn = nn.CrossEntropyLoss()
    
    def _check_inputs(self, images):
        if not (0 <= images.min() and images.max() <= 1):
            raise ValueError("Input images must be in the range [0, 1].")

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        self.model.eval()
        self._check_inputs(images)
        loss_fn = nn.CrossEntropyLoss()
        adv_images = images.clone()
        # print(f"[DEBUG] adv_images.min: {adv_images.min()}, adv_images.max: {adv_images.max()}")
        total_data = images.size(0)  # 第一个维度表示样本数量
        print(f"Total number of data points: {total_data}")
        CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
        CIFAR_STD = [0.2023, 0.1994, 0.2010]
        normalize = transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        iternum = torch.zeros(images.size(0), dtype=torch.int32, device=images.device)  # 初始化张量存储迭代次数
        success_count = 0  # 初始化成功生成对抗样本的计数器
        for i in range(images.size(0)):
            adv_image = adv_images[i:i+1].clone()  # 当前样本
            label = labels[i].item()
            success = False
            for step in range(self.steps):
                adv_image.requires_grad = True
                trans_image= normalize(adv_image)
                # print(f"[DEBUG] trans_image.min: {trans_image.min()}, adv_image.max: {trans_image.max()}")
                outputs = self.model(trans_image)
                # outputs = self.model(adv_image)
                output = outputs[0:1]
                target_label = labels[i:i+1]
                # Calculate loss
                if self.targeted:
                    cost = -loss_fn(output, target_label)
                else:
                    cost = loss_fn(output, target_label)

                # Update adversarial images
                grad = torch.autograd.grad(cost, adv_image,
                                        retain_graph=False, create_graph=False)[0]

                adv_image = adv_image.detach() + self.alpha*grad.sign()
                delta = torch.clamp(adv_image - images[i:i+1], min=-self.eps, max=self.eps)
                adv_image = torch.clamp(images[i:i+1] + delta, min=0, max=1).detach()
                #检查分类是否改变
                pred = output.argmax(dim=1).item()
                if pred != label:  # 非 targeted：只需预测类别不同即可
                    iternum[i] = step + 1
                    success = True
                    success_count += 1  # 成功计数器 +1
                    break
            if not success:
                # 如果达到最大迭代次数仍未生成成功
                iternum[i] = self.steps
        print(f"Number of successfully generated adversarial samples: {success_count}/{total_data}")
        return iternum 

def pgd_attack(model, images, labels, eps=3/255, alpha=0.001, steps=50, random_start=True, targeted=False):
    """
    使用 PGD 生成对抗性样本。
    """
    PGD_attack = PGDAttack(
        model=model,
        eps=eps,
        alpha=alpha,
        steps=steps,
        random_start=random_start,
        targeted=targeted
    )
    iternum = PGD_attack.forward(images, labels)
    return iternum

