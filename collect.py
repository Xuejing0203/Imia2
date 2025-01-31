import argparse
import random
from tqdm import tqdm as tq
import sys
from collections import OrderedDict
import torch
from torchvision import datasets, transforms
import torchvision.transforms as trans
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import grad
from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
from utils import rescale, apply_normalization, get_preds, Softmax, advDistance, advAttack, simpleBlackAttack, to_one_hot, Entropy, ModEntropy, square_attack_linf
from pathlib import Path
import importlib
import os
import sys
sys.path.append('pytorch-cifar')
import models

parser = argparse.ArgumentParser(description='Apply different strategies for MIA to target model.')

parser.add_argument('--seed', type=int, help='Set random seed for reproducibility.')
parser.add_argument('--dataset', type=str, default='cifar10', help='Which dataset to use for the experiments.')
parser.add_argument('--model_type', type=str, help='Model Architecture to attack.')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size for batched computations.')
parser.add_argument('--output_dir', type=str, default='./', help='Where to store output data.')
parser.add_argument('--data_dir', type=str, default='./data', help='Where to retrieve the dataset.')
parser.add_argument('--trained_dir', type=str, default='./trained_models', help='Where to retrieve trained models.')
parser.add_argument('--dry_run', action='store_true', default=False, help='Test run on 100 samples.')

exp_parameters = parser.parse_args()
cuda = True if torch.cuda.is_available() else False
# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Loading model
print("loading model........")
model_dir = exp_parameters.trained_dir
if exp_parameters.dataset == 'cifar10':
    model_dir = model_dir + '/cifar10'
model_type = exp_parameters.model_type

model = getattr(models, model_type)().cuda()
model = torch.nn.DataParallel(model)
checkpoint = torch.load("./pytorch-cifar/checkpoint_resnet_cifar10/ckpt.pth")
model.load_state_dict(checkpoint['net'])
model.eval()

# Record time of computation
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()

# Setting seed for reproducibility
seed = exp_parameters.seed

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Loading Datasets
print('loading datasets..............')
test_TRANSFORM = trans.Compose([
    # transforms.Resize(32),
    trans.ToTensor()
    ])
train_TRANSFORM = trans.Compose([
    # transforms.Resize(32),
    trans.ToTensor()
    ])

batch_size = exp_parameters.batch_size

data_dir = exp_parameters.data_dir
# train_dataset = datasets.STL10(data_dir, split='train', download=True, transform=train_TRANSFORM)
# test_dataset = datasets.STL10(data_dir, split='test', download=True, transform=test_TRANSFORM)
train_dataset = datasets.CIFAR10(data_dir, train=True, download=False, transform=train_TRANSFORM)
test_dataset = datasets.CIFAR10(data_dir, train=False, download=False, transform=test_TRANSFORM)

train_set = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
images1, labels1 = next(iter(train_set))
print(f"[DEBUG] images1.min: {images1.min()}, images1.max: {images1.max()}")

test_set = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
images0, labels0 = next(iter(test_set))


if cuda:
    images1 = images1.cuda()
    images0 = images0.cuda()
    labels1 = labels1.cuda()
    labels0 = labels0.cuda()

bs = 500
total_train_dataset = TensorDataset(images1, labels1)    #训练集是1，测试集是0
total_test_dataset = TensorDataset(images0, labels0)
total_train_loader = DataLoader(total_train_dataset, batch_size=bs, shuffle = True)
total_test_loader = DataLoader(total_test_dataset, batch_size=bs, shuffle = True)

Loss = torch.nn.CrossEntropyLoss(reduction='none')
if cuda:
    model.cuda()
    Loss.cuda()


def apply_gaussian_blur(batch_images, blur_radius=0.7):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: [transforms.ToPILImage()(img) for img in x]),  # 转换为 PIL 图像
        transforms.Lambda(lambda imgs: [
            img.filter(ImageFilter.GaussianBlur(radius=blur_radius)) for img in imgs]),  # 高斯模糊
        transforms.Lambda(lambda imgs: torch.stack([transforms.ToTensor()(img) for img in imgs])),  # 转回 Tensor
    ])
    
    blurred_images = transform(batch_images)
    return blurred_images

# Results will be save in a csv file 
scoreLists0 = [] #保存测试集数据
scoreLists1 = [] #保存训练集数据
softmax = torch.nn.Softmax(dim=1)
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
normalize = transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
for batch_images0, batch_labels0 in total_test_loader:
    batch_images0 = batch_images0.cuda()
    batch_images0 = apply_gaussian_blur(batch_images0, blur_radius=0.7)
    print(f"[DEBUG] test_images0.min: {batch_images0.min()}, images0.max: {batch_images0.max()}")
    batch_labels0 = batch_labels0.cuda()
    logits0 = model(apply_normalization(batch_images0, 'cifar'))
  
    #softmax attack
    softmax_scores0 = torch.max(softmax(logits0), 1)[0].cpu().data.numpy()

    #计算entropy
    softprob0 = softmax(logits0).detach()
    Entr0 = -Entropy(softprob0).cpu().data.numpy()

    #计算modified entropy 
    ModEntr0 = -ModEntropy(softprob0, batch_labels0).cpu().data.numpy()
    #计算loss
    Loss = torch.nn.CrossEntropyLoss(reduction='none')
    loss0 = -Loss(logits0, batch_labels0).detach().cpu().data.numpy()
    # # #计算pgd attack
    pgd_iternum0 = advAttack(model, batch_images0, batch_labels0, batch_size = batch_size, eps=3/255, alpha=0.001, steps=50, random_start=True, targeted=False).cpu().data.numpy()
    # # #计算simblack attack
    
    simpleblack_nums0 = simpleBlackAttack(model, batch_images0, batch_labels0, batch_size = batch_size, max_iters=300, freq_dims=32, stride=7, epsilon=0.05, linf_bound=0.0, order='rand', targeted=False, pixel_attack=False, log_every=40 ).cpu().data.numpy()
    #合并结果
    # batch_scores0 = np.stack([softmax_scores0, Entr0, ModEntr0], axis=1)
    batch_scores0 = np.stack([softmax_scores0, Entr0, ModEntr0, loss0, pgd_iternum0, simpleblack_nums0], axis=1)
    scoreLists0.append(batch_scores0)
scores0 = np.concatenate(scoreLists0)
print('testloader computation: done......................................................................')

for batch_images1, batch_labels1 in total_train_loader:
    batch_images1 = batch_images1.cuda()
    batch_labels1 = batch_labels1.cuda()
    logits1 = model(apply_normalization(batch_images1, 'cifar'))

    #计算softmax attack
    softmax_scores1 = torch.max(softmax(logits1), 1)[0].cpu().data.numpy()
    #计算entropy
    softprob1 = softmax(logits1).detach()
    Entr1 = -Entropy(softprob1).cpu().data.numpy()
    #计算modified entropy 
    ModEntr1 = -ModEntropy(softprob1, batch_labels1).cpu().data.numpy()
    #计算loss
    Loss = torch.nn.CrossEntropyLoss(reduction='none')
    loss1 = -Loss(logits1, batch_labels1).detach().cpu().data.numpy()
    #计算pgd attack
    pgd_iternum1 = advAttack(model, batch_images1, batch_labels1, batch_size = batch_size, eps=3/255, alpha=0.001, steps=50, random_start=True, targeted=False).cpu().data.numpy()
    # #计算simblack attack
    simpleblack_nums1 = simpleBlackAttack(model, batch_images1, batch_labels1, batch_size = batch_size, max_iters=300, freq_dims=32, stride=7, epsilon=0.05, linf_bound=0.0, order='rand', targeted=False, pixel_attack=False, log_every=40 ).cpu().data.numpy()
  
    #合并结果
    batch_scores1 = np.stack([softmax_scores1, Entr1, ModEntr1, loss1, pgd_iternum1, simpleblack_nums1], axis=1)
    scoreLists1.append(batch_scores1)
scores1 = np.concatenate(scoreLists1)
print('trainloader computation: done')

end.record()
torch.cuda.synchronize()
print('Elapsed time of computation in miliseconds: %f' % (start.elapsed_time(end)))
df_scores0 = pd.DataFrame(scores0, columns=['Softmax Response', 'Entropy', 'Modified Entropy','Loss','PGD Attack', 'Simpleblack Attack'])
df_scores1 = pd.DataFrame(scores1, columns=['Softmax Response', 'Entropy', 'Modified Entropy', 'Loss','PGD Attack', 'Simpleblack Attack'])
outdir = Path(exp_parameters.output_dir)
results_dir = outdir / 'RawResults'
results_dir.mkdir(parents=True, exist_ok=True)

df_scores0.to_csv(results_dir / f'scores0_{model_type}_partial.csv', index=False)
df_scores1.to_csv(results_dir / f'scores1_{model_type}_partial.csv', index=False)
