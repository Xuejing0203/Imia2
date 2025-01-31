import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from utils import rescale01, computeMetrics, computeBestThreshold, evalBestThreshold, hyperMetrics

parser = argparse.ArgumentParser(description='Analyse criteria obtained from different MIAs.')

parser.add_argument('--model_type', type=str, help='Model Architecture to attack.')
parser.add_argument('--num_iters', type=int, default=20, help='Number of iterations for empirical estimation.')
parser.add_argument('--mode', type=int, help='What part of the analysis to compute.')
parser.add_argument('--working_dir', type=str, default='./', help='Where to collect and store data.')

exp_parameters = parser.parse_args()

currdir = exp_parameters.working_dir
if not os.path.exists(currdir + '/CompleteResults_resnet_cifar10'):
    os.makedirs(currdir + '/CompleteResults_resnet_cifar10')

num_runs_for_random = exp_parameters.num_iters
model_type = exp_parameters.model_type
mode = exp_parameters.mode
scores0 = pd.read_csv('./RawResults_resnet_cifar10/RawResults/scores0_ResNet50_partial.csv')       
scores1 = pd.read_csv('./RawResults_resnet_cifar10/RawResults/scores1_ResNet50_partial.csv')        #score1保存训练集数据
FPR = np.linspace(0, 1, num=1001) 


if mode == 1:
    try:
        dfMetricsBalanced = pd.read_csv(currdir + '/CompleteResults_resnet_cifar100_labelonly_half/BalancedMetrics_' + model_type + '.csv')
        dfTPRBalanced = pd.read_csv(currdir + '/CompleteResults_resnet_cifar100_labelonly_half/BalancedROC_' + model_type + '.csv')
    except FileNotFoundError:
        dfMetricsBalanced = pd.DataFrame(columns=['Attack Strategy',
                                                  'AUROC', 'AUROC STD',
                                                  'Best Accuracy', 'Best Accuracy STD',
                                                  'FPR at TPR80', 'FPR at TPR80 STD',
                                                  'FPR at TPR85', 'FPR at TPR85 STD',
                                                  'FPR at TPR90', 'FPR at TPR90 STD',
                                                  'FPR at TPR95', 'FPR at TPR95 STD']) 
        dfTPRBalanced = pd.DataFrame(FPR, columns=['FPR'])
    for column_name in scores0:
    # for column_name in [3]:
        aux_list_metrics = [] 
        aux_list_TPR = []     

        criteria0 = scores0[[column_name]].values 
        criteria1 = scores1[[column_name]].values

        for i in range(num_runs_for_random): 
            # Setting Random Seed
            np.random.seed(i)

            indx_eval0 = np.random.choice(criteria0.shape[0], size=3000, replace=False) 
            indx_eval1 = np.random.choice(criteria1.shape[0], size=3000, replace=False)
            criteria0_eval = criteria0[indx_eval0]
            criteria1_eval = criteria1[indx_eval1]

            TPR_, metrics_ = computeMetrics(criteria0_eval, criteria1_eval, FPR)
            aux_list_metrics.append(metrics_) 
            aux_list_TPR.append(TPR_)

        metrics = np.stack(aux_list_metrics, 1)
        mean_metrics = np.mean(metrics, 1) 
        std_metrics = np.std(metrics, 1)

        new_row = {"Attack Strategy": column_name,
                   'AUROC': mean_metrics[0], 'AUROC STD': std_metrics[0],
                   'Best Accuracy': mean_metrics[1], 'Best Accuracy STD': std_metrics[1],
                   'FPR at TPR80': mean_metrics[2], 'FPR at TPR80 STD': std_metrics[2],
                   'FPR at TPR85': mean_metrics[3], 'FPR at TPR85 STD': std_metrics[3],
                   'FPR at TPR90': mean_metrics[4], 'FPR at TPR90 STD': std_metrics[4],
                   'FPR at TPR95': mean_metrics[5], 'FPR at TPR95 STD': std_metrics[5]} #创建新行

        # dfMetricsBalanced = dfMetricsBalanced.append(new_row, ignore_index=True)
        dfMetricsBalanced = pd.concat([dfMetricsBalanced, pd.DataFrame([new_row])], ignore_index=True) 
        TPR = np.stack(aux_list_TPR, 1) 
        mean_TPR = np.mean(TPR, 1)
        std_TPR = np.std(TPR, 1)  

        dfTPRaux = pd.DataFrame(np.stack((mean_TPR, std_TPR), axis=1),
                                columns=[column_name + ' TPR mean', column_name + ' TPR std'])
        # dfTPRaux = pd.DataFrame(np.stack((mean_TPR, std_TPR), axis=1),
        #                         columns=[str(column_name) + ' TPR mean', str(column_name) + ' TPR std'])
        dfTPRBalanced = dfTPRBalanced.join(dfTPRaux) #将 dfTPRaux 加入到 dfTPRBalanced


    dfMetricsBalanced.to_csv(currdir + '/CompleteResults_resnet_cifar10/BalancedMetrics_' + model_type + '.csv', index=False)
    dfTPRBalanced.to_csv(currdir + '/CompleteResults_resnet_cifar10/BalancedROC_' + model_type + '.csv', index=False)
