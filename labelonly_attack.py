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

    
def initialize(model, images, params):
    success = False
    num_evals = 0
    images = images.cuda()
    # Find a misclassified random noise.
    # 随机噪声生成
    while num_evals < 1e4:
        random_noise = torch.empty(params['shape']).uniform_(params['clip_min'], params['clip_max']).cuda()
        # 判断当前噪声是否能够导致错误分类
        success = decision_function(model, random_noise.unsqueeze(0), params) 
        num_evals += 1
        if success.any():
            break
    if not success.any():  # Initialization failed after maximum evaluations
        return images, False
    # 二分法最小化与原图的L2距离
    low = 0.0
    high = 1.0
    while high - low > 0.001:
        mid = (high + low) / 2.0
        blended = (1 - mid) * images + mid * random_noise
        success = decision_function(model, blended.unsqueeze(0), params)
        if success:
            high = mid
        else:
            low = mid
    initialization = (1 - high) * images + high * random_noise
    return initialization, True

def decision_function(model, images, params):
    images = torch.clamp(images, params['clip_min'], params['clip_max'])
    with torch.no_grad():
        images = images.view(-1, 3, 32, 32) # 调整形状为 [batch_size * n, 3, 32, 32]
        # print(images.shape)
        prob = model(utils.apply_normalization(images, 'cifar'))
    predicted_labels = torch.argmax(prob, dim=1)
    return (predicted_labels == params['original_label']).float() 

def compute_distance(x_ori, x_pert, constraint = 'l2'):
    if constraint == 'l2':
        # L2 distance: Frobenius norm (Euclidean distance)
        return torch.norm(x_ori - x_pert, p=2)
    elif constraint == 'linf':
        # L∞ distance: maximum absolute difference
        return torch.max(torch.abs(x_ori - x_pert))
    else:
        raise ValueError("Unknown constraint: {}".format(constraint))

def project(original_image, perturbed_images, alphas, params):
    # Reshape alphas to match the dimensions of perturbed_images
    alphas_shape = [len(alphas)] + [1] * len(params['shape'])
    alphas = alphas.view(alphas_shape).cuda()  # PyTorch's view method replaces reshape
    original_image = original_image.cuda()
    perturbed_images = perturbed_images.cuda()
    if params['constraint'] == 'l2':
        # L2 projection: Blend original and perturbed images using alphas
        return (1 - alphas) * original_image + alphas * perturbed_images
    
    elif params['constraint'] == 'linf':
        # L∞ projection: Clip perturbed images within a range around the original image
        lower_bound = original_image - alphas
        upper_bound = original_image + alphas
        return torch.clamp(perturbed_images, lower_bound, upper_bound)
    else:
        raise ValueError("Unknown constraint: {}".format(params['constraint']))
    
def binary_search_batch(original_image, perturbed_images, model, params):
    # Compute distance between each perturbed image and the original image.
    dists_post_update = torch.tensor([
        compute_distance(original_image, perturbed_image, params['constraint']) 
        for perturbed_image in perturbed_images
    ])
    # Choose upper thresholds in binary search based on constraint.
    if params['constraint'] == 'linf':
        highs = dists_post_update
        # Stopping criteria
        thresholds = torch.minimum(dists_post_update * params['theta'], torch.tensor(params['theta']))
    else:
        highs = torch.ones(len(perturbed_images)).cuda()
        thresholds = torch.tensor(params['theta']).cuda()
    lows = torch.zeros(len(perturbed_images)).cuda()
    # Call recursive function.
    while torch.max((highs - lows) / thresholds) > 1:
        # Projection to mids
        mids = (highs + lows) / 2.0
        mid_images = project(original_image, perturbed_images, mids, params)       
        # Update highs and lows based on model decisions
        decisions = decision_function(model, mid_images, params).cuda()
        lows = torch.where(decisions == 0, mids, lows)
        highs = torch.where(decisions == 1, mids, highs)
    out_images = project(original_image, perturbed_images, highs, params)
    # Compute distance of the output image to select the best choice.
    dists = torch.tensor([
        compute_distance(original_image, out_image, params['constraint'])
        for out_image in out_images
    ])
    idx = torch.argmin(dists)
    dist = dists_post_update[idx]
    out_image = out_images[idx]

    return out_image, dist

def select_delta(params, dist_post_update):
    if params['cur_iter'] == 1:
        # For first iteration, delta is a constant value based on clip_max and clip_min
        delta = 0.1 * (params['clip_max'] - params['clip_min'])
    else:
        # Compute delta based on the constraint
        dist_post_update = torch.tensor(dist_post_update, dtype=torch.float32)

        if params['constraint'] == 'l2':
            delta = torch.sqrt(torch.tensor(params['d'], dtype=torch.float32)) * params['theta'] * dist_post_update
        elif params['constraint'] == 'linf':
            delta = params['d'] * params['theta'] * dist_post_update
    return delta

def approximate_gradient(model, sample, num_evals, delta, params):
    clip_max, clip_min = params['clip_max'], params['clip_min']
    # Generate random vectors.
    noise_shape = [num_evals] + list(params['shape'])
    if params['constraint'] == 'l2':
        rv = torch.randn(*noise_shape, dtype=torch.float32)
    elif params['constraint'] == 'linf':
        rv = torch.empty(noise_shape, dtype=torch.float32).uniform_(-1, 1)
    
    # Normalize random vectors (l2 normalization).
    rv = rv / torch.sqrt(torch.sum(rv ** 2, dim=(1, 2, 3), keepdim=True))

    # Perturb the image.
    perturbed = sample.cpu() + delta * rv
    perturbed = torch.clamp(perturbed, clip_min, clip_max).cuda()
    rv = (perturbed - sample) / delta

    # Query the model.
    decisions = decision_function(model, perturbed, params)
    decision_shape = [len(decisions)] + [1] * len(params['shape'])
    fval = 2 * decisions.float().reshape(decision_shape) - 1.0

    # Baseline subtraction (when fval differs)
    if torch.mean(fval) == 1.0:  # Label changes.
        gradf = torch.mean(rv, dim=0)
    elif torch.mean(fval) == -1.0:  # Label does not change.
        gradf = -torch.mean(rv, dim=0)
    else:
        fval -= torch.mean(fval)
        gradf = torch.mean(fval * rv, dim=0)

    # Normalize the gradient direction.
    gradf = gradf / torch.norm(gradf)
    return gradf

def geometric_progression_for_stepsize(x, update, dist, model, params):
    epsilon = dist / torch.sqrt(torch.tensor(params['cur_iter'], dtype=torch.float32))
    def phi(epsilon):
        # Compute the perturbed image
        new = x + epsilon * update
        # Check if the perturbation is successful
        success = decision_function(model, new.unsqueeze(0), params)
        return success

    # Decrease epsilon geometrically until the desired side of the boundary is reached
    while not phi(epsilon):
        epsilon /= 2.0

    return epsilon

def _check_inputs(images):
    if not (0 <= images.min() and images.max() <= 1):
        raise ValueError("Input images must be in the range [0, 1].")

def hsja(model, images, labels, clip_min, clip_max, constraint='l2', num_iterations=50, gamma=1.0, stepsize_search='geometric_progression', max_num_evals=1e4, init_num_evals=100, verbose=True):
    r"""
    Overridden.
    """
    model.eval()
    _check_inputs(images)
    num_classes = 100
    images = images.cuda()
    labels = labels.cuda()
    with torch.no_grad():
        output = model(utils.apply_normalization(images, 'cifar'))
        predicted_label = torch.argmax(output, dim=1)
    
    # 记录哪些样本是正确预测的
    correct_predictions = predicted_label == labels
    # print(f"Number of correctly predicted samples: {torch.sum(correct_predictions).item()}")
    # # 对每个正确预测的样本，初始化 original_label 为其正确标签
    # original_labels = labels.clone()
    # original_labels[~correct_predictions] = -1  # 将错误预测的样本的标签设为 -1（标记为无效）
    
    # 记录正确预测样本的原始标签
    correct_labels = labels[correct_predictions]
    correct_images = images[correct_predictions]
    correct_images_by_class = [correct_images[correct_labels == c] for c in range(num_classes)]

    # print(images.shape)
    # 记录每个样本的迭代次数
    iteration_counts = torch.zeros(images.shape[0]).long().cuda()

    for i in range(images.shape[0]): 
        single_image = images[i].unsqueeze(0)
        print(single_image.shape) 
        with torch.no_grad():
            output = model(utils.apply_normalization(single_image, 'cifar'))
            predict_label = torch.argmax(output, dim=1)
            # print(f"Predicted label: {predict_label.item()}")
        # 如果预测错误，则跳过该样本，直接将迭代次数设为0
        if predict_label != labels[i]:
            iteration_counts[i] = 0
            continue  # 不进行扰动初始化和对抗攻击
        
        # 确保新标签与模型预测的标签不同
        fake_label = np.random.choice([j for j in range(num_classes) if j != predict_label.item()])
        if len(correct_images_by_class[fake_label]) > 0:
            fake_image = correct_images_by_class[fake_label][torch.randint(0, len(correct_images_by_class[fake_label]), (1,))]
        else:
            for _ in range(num_classes):
                fake_label = np.random.choice([j for j in range(num_classes) if j != predict_label.item()])
                if len(correct_images_by_class[fake_label]) > 0:
                    fake_image = correct_images_by_class[fake_label][torch.randint(0, len(correct_images_by_class[fake_label]), (1,))]
                    break
        # print(f"fake label: {fake_label.item()}")
        params = {
            'clip_max': 1.0,
            'clip_min': 0.0,
            'shape': single_image.shape,
            'original_label': fake_label,
            'constraint': 'l2',
            'num_iterations': 100,                  #total number of model evaluations for the entire algorithm
            'gamma': 1.0,                          #用于设置二分查找阈值的参数，影响攻击的强度。
            'd': int(np.prod(fake_image.shape)),       # 计算d（样本的总元素数）
            'stepsize_search': 'geometric_progression',
            'max_num_evals': 1e4,                  # maximum number of evaluations for estimating gradient (for each iteration)
            'init_num_evals': 100,
            'verbose': True
        }
        # 根据约束类型设置theta (L2约束)
        params['theta'] = params['gamma'] / (np.sqrt(params['d']) * params['d'])
        # perturbed, success = initialize(model, single_image, params)
        # if not success:  # If initialization fails, skip this sample
        #     print(f"Initialization failed for sample {i}. Skipping to next sample.")
        #     iteration_counts[i] = params['num_iterations']  # Record max iterations for this sample
        #     continue

        perturbed = fake_image
        for j in range(params['num_iterations']):
            # 初始化扰动
            # Project the initialization to the boundary.
            params['cur_iter'] = j + 1 
            perturbed, dist_post_update = binary_search_batch(single_image, perturbed.unsqueeze(0), model, params)
            dist = compute_distance(perturbed, single_image, constraint)

            # Choose delta
            delta = select_delta(params, dist_post_update)            
            # Choose number of evaluations选择评估次数
            num_evals = int(params['init_num_evals'] * torch.sqrt(torch.tensor(j + 1, dtype=torch.float32)))
            num_evals = min(num_evals, params['max_num_evals'])
            # Approximate gradient近似梯度
            gradf = approximate_gradient(model, perturbed, num_evals, delta, params)     
            if params['constraint'] == 'linf':
                update = torch.sign(gradf)
            else:
                update = gradf
            # Search for step size搜索步长
            if params['stepsize_search'] == 'geometric_progression':
                # Find step size查找步长
                epsilon = geometric_progression_for_stepsize(perturbed, update, dist, model, params)  
                # Update the sample更新样本
                perturbed = torch.clamp(perturbed + epsilon * update, params['clip_min'], params['clip_max'])
                # Binary search to return to the boundary二分查找返回边界
                perturbed, dist_post_update = binary_search_batch(single_image, perturbed.unsqueeze(0), model, params)
            # compute new distance.计算新的距离
            dist = compute_distance(perturbed, single_image, constraint)
            # 检查是否生成了对抗性样本
            output = model(utils.apply_normalization(perturbed, 'cifar'))
            new_label = torch.argmax(output, dim=1).item()
            # print(f"new label: {new_label.item()}")
            # 如果模型预测发生变化，说明生成了对抗性样本
            if new_label == params['original_label'] and dist < 0.5:
                iteration_counts[i] = j + 1  # 记录生成对抗性样本的迭代次数
                if verbose:
                    print(f"Sample {i} successfully generated adversarial sample at iteration {j+1}.")
                break  # 跳出当前样本的迭代，继续处理下一个样本
        
        # 如果对该样本没有成功生成对抗样本，可以给它一个默认值（如0或者max_num_iterations）
        if iteration_counts[i] == 0:  # 这个样本没有生成对抗性样本
            iteration_counts[i] = params['num_iterations']
        if verbose:
            print('iteration: {:d}, {:s} distance {:.4E}'.format(j+1, constraint, dist))
    return perturbed, iteration_counts

def hsja_attack(model, images, labels, clip_max = 1, clip_min = 0, constraint = 'l2', num_iterations = 100, gamma = 1.0, stepsize_search = 'geometric_progression', max_num_evals = 1e4, init_num_evals = 100):
    """
    使用 hsja 生成对抗性样本。
    """
    
    _, iternum = hsja(model, images, labels, clip_max = 1, clip_min = 0, constraint = 'l2', num_iterations = 100, gamma = 1.0, stepsize_search = 'geometric_progression', max_num_evals = 1e4, init_num_evals = 100)
    return iternum

#######统计距离，dist<0.05
