# from auto_attack.autoattack.autoattack import AutoAttack
import torch
from torch.autograd import grad
import numpy as np
from PGD_attack import pgd_attack
from simpleblackattack import simpleblackbox
from labelonly_attack import hsja_attack
from un_labelonly_attack import un_hsja_attack
from labelonly2017_attack import main_attack
# from square_attack import square_attack
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.metrics import roc_curve, accuracy_score
from scipy.fftpack import dct, idct
import torchvision.transforms as trans
import math

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR_SIZE = 32
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
CIFAR_TRANSFORM = trans.Compose([
    trans.ToTensor()])

def rescale(tensor, max_, min_):
    """ Rescale a pytorch tensor to [0,1].
    tensor: pytorch tensor with dimensions [batch,channels,width,height]
    max_: pytorch tensor containing the maximum values per channel
    min_: pytorch tensor containing the minimum values per channel
    outputs -> rescaled pytorch tensor with the same dimensions as 'tensor'
    """
    max_ = max_.reshape(1, -1, 1, 1)
    min_ = min_.reshape(1, -1, 1, 1)
    return (tensor - min_) / (max_ - min_ + 1e-8)


def unscale(tensor, max_, min_):
    """ Rescale a pytorch tensor back to its original values.
    tensor: pytorch tensor with dimensions [batch,channels,width,height]
    max_: pytorch tensor containing the maximum values per channel
    min_: pytorch tensor containing the minimum values per channel
    outputs -> rescaled pytorch tensor with the same dimensions as 'tensor'
    """
    max_ = max_.reshape(1, -1, 1, 1)
    min_ = min_.reshape(1, -1, 1, 1)
    return tensor * (max_ - min_) + min_

# applies the normalization transformations
def apply_normalization(imgs, dataset):
    if dataset == 'imagenet':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    elif dataset == 'cifar':
        mean = CIFAR_MEAN
        std = CIFAR_STD
    elif dataset == 'mnist':
        mean = MNIST_MEAN
        std = MNIST_STD
    else:
        mean = [0, 0, 0]
        std = [1, 1, 1]
    imgs_tensor = imgs.clone()
    if dataset == 'mnist':
        imgs_tensor = (imgs_tensor - mean[0]) / std[0]
    else:
        if imgs.dim() == 3:
            for i in range(imgs_tensor.size(0)):
                imgs_tensor[i, :, :] = (imgs_tensor[i, :, :] - mean[i]) / std[i]
        else:
            for i in range(imgs_tensor.size(1)):
                imgs_tensor[:, i, :, :] = (imgs_tensor[:, i, :, :] - mean[i]) / std[i]
    return imgs_tensor
# get most likely predictions and probabilities for a set of inputs
def get_preds(model, inputs, dataset_name, correct_class=None, batch_size=25, return_cpu=True):
    num_batches = int(math.ceil(inputs.size(0) / float(batch_size)))
    softmax = torch.nn.Softmax()
    all_preds, all_probs = None, None
    transform = trans.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    for i in range(num_batches):
        upper = min((i + 1) * batch_size, inputs.size(0))
        input = apply_normalization(inputs[(i * batch_size):upper], dataset_name)
        # print(f"[DEBUG] input.min: {input.min()}, images.max: {input.max()}")
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        output = softmax.forward(model.forward(input_var))
        if correct_class is None:
            prob, pred = output.max(1)
        else:
            prob, pred = output[:, correct_class], torch.autograd.Variable(torch.ones(output.size()) * correct_class)
        if return_cpu:
            prob = prob.data.cpu()
            pred = pred.data.cpu()
        else:
            prob = prob.data
            pred = pred.data
        if i == 0:
            all_probs = prob
            all_preds = pred
        else:
            all_probs = torch.cat((all_probs, prob), 0)
            all_preds = torch.cat((all_preds, pred), 0)
    return all_preds, all_probs


def to_one_hot(y, num_classes):
    """ Convert a list of indexes into a list of one-hot encoded vectors.
    y: pytorch tensor of shape [batch], containing a list of integer labels
    num_classes: int corresponding to the number of classes
    outputs -> pytorch tensor containing the one-hot encoding of the labels 'y'. Its dimensions are [batch,num_classes]
    """
    if len(y.shape) == 1:
        y = torch.unsqueeze(y, 1)
    y_one_hot = torch.zeros(y.shape[0], num_classes)
    if cuda:
        y_one_hot = y_one_hot.cuda()
    y_one_hot = y_one_hot.scatter(1, y, 1)
    return y_one_hot

def hyperMetrics(scores0, scores1, FPR):
    labels0 = np.zeros_like(scores0)  # 测试集（非成员），所有值为 0
    labels1 = np.ones_like(scores1)
    
    
    predicted_labels0 = (scores0 >= 26).astype(int)  # 大于等于 100 的被认为是成员（1），否则为非成员（0）
    predicted_labels1 = (scores1 >= 26).astype(int)
    predicted_labels = np.concatenate((predicted_labels0, predicted_labels1))  # 合并分数
    labels = np.concatenate((labels0, labels1))

    # ROC curve
    fpr, tpr, thr = roc_curve(labels, predicted_labels)
    TPR = np.interp(FPR, fpr, tpr)

    # FPR @TPR 80, 85, 90, 95
    metrics = np.interp([.80, .85, .90, .95], tpr, fpr)

    # AUROC
    # AUROC = roc_auc(labels, scores)
    AUROC = np.trapz(tpr, fpr) 
    # Optimal Accuracy
    # AccList = [accuracy_score(scores >= t, labels) for t in thr]
    AccList = [accuracy_score(predicted_labels, labels) for t in thr]
    Acc_opt = np.max(AccList)

    # Combine metrics into a single array
    metrics = np.append((AUROC, Acc_opt), metrics)

    return TPR, metrics

def computeMetrics(scores0, scores1, FPR):
    """ Computes performance metrics using the scores computed with a certain strategy. The performance scores computed
    include the AUROC score, the best accuracy achieved for any threshold and the FPR at TPR 95%. Also computes the TPR
    values corresponding to the FPR values given as input.
    scores0: numpy array containing the negative scores
    scores1: numpy array containing the positive scores
    FPR: numpy array. TPR values will be interpolated for this FPR values
    outputs -> tuple of ('TPR', 'metrics'). 'TPR' is a numpy array. 'metrics' is a list containing the AUROC score, best
    accuracy, FPR at TPR 80, 85, 90 and 95%
    """
    labels0 = np.zeros_like(scores0)
    labels1 = np.ones_like(scores1)

    scores = np.concatenate((scores0, scores1))
    labels = np.concatenate((labels0, labels1))

    # ROC curve
    fpr, tpr, thr = roc_curve(labels, scores)
    TPR = np.interp(FPR, fpr, tpr)

    # FPR @TPR95

    metrics = np.interp([.80, .85, .90, .95], tpr, fpr)

    # AUROC

    AUROC = roc_auc(labels, scores)

    # Optimal Accuracy

    AccList = [accuracy_score(scores > t, labels) for t in thr]
    Acc_opt = np.max(AccList)

    metrics = np.append((AUROC, Acc_opt), metrics)

    return TPR, metrics

def computeMetricsAlt(scores, labels, FPR):
    """
    Computes several performance metrics based on scores.

    scores: Numpy array. scores for both classes with corresponding labels given in 'labels'.
    labels: Numpy array. labels for 'scores'.
    FPR: array indicating FPR values for the ROC curve.
    """

    # AUROC

    AUROC = roc_auc(labels, scores)
    recompute_AUROC = False

    if AUROC < .5:
        scores = -scores
        recompute_AUROC = True

    if recompute_AUROC:
        AUROC = roc_auc(labels, scores)

    # ROC curve
    fpr, tpr, thr = roc_curve(labels, scores, drop_intermediate=True)
    interpolator = interp1d(fpr, tpr, kind='previous')
    TPR = interpolator(FPR)

    # FPR @TPR95

    metrics = interpolator([.80, .85, .90, .95])

    # Optimal Accuracy

    AccList = (tpr + 1.0 - fpr) / 2
    Acc_opt = AccList.max()

    metrics = np.append((AUROC, Acc_opt), metrics)

    return TPR, metrics

def computeBestThreshold(scores0, scores1):
    """ Computes the threshold which maximizes the accuracy given the scores.
        scores0: numpy array containing the negative scores
        scores1: numpy array containing the positive scores
        outputs -> thresh achieving the best accuracy for the input scores. Float
        """
    labels0 = np.zeros_like(scores0)
    labels1 = np.ones_like(scores1)

    scores = np.concatenate((scores0, scores1))
    labels = np.concatenate((labels0, labels1))

    # ROC curve
    _, _, thr = roc_curve(labels, scores)

    AccList = [accuracy_score(scores0 > t, labels0) + accuracy_score(scores1 > t, labels1) for t in thr]
    Acc_opt_indx = np.argmax(AccList)

    return thr[Acc_opt_indx]


def evalBestThreshold(thr_opt, scores0, scores1):
    """ Computes the balanced accuracy and FPR of the given scores with the given threshold.
    thr_opt: float threshold
    scores0: numpy array containing the negative scores
    scores1: numpy array containing the positive scores
    outputs -> tuple containing the balanced accuracy and FPR

    """
    labels0 = np.zeros_like(scores0)
    labels1 = np.ones_like(scores1)

    Acc = (accuracy_score(scores0 > thr_opt, labels0) + accuracy_score(scores1 > thr_opt, labels1)) / 2

    FPR = sum(scores0 > thr_opt) / len(scores0)

    return [Acc, FPR[0]]

def evalThresholdAlt(thr_opt, scores, labels):
    indx0 = np.nonzero(labels==0)
    indx1 = np.nonzero(labels==1)
    
    scores0 = scores[indx0]
    scores1 = scores[indx1]
    
    labels0 = np.zeros_like(scores0)
    labels1 = np.ones_like(scores1)

    Acc = (accuracy_score(scores0 > thr_opt, labels0) + accuracy_score(scores1 > thr_opt, labels1)) / 2

    FPR = sum(scores0 > thr_opt) / len(scores0)

    return np.asarray([Acc, FPR],dtype=np.float64)

def Softmax(in_tensor):
    """ Apply Softmax to the input tensor. 
    in_tensor: pytorch tensor with dimensions [batch, length]
    outputs -> pytorch tensor with the same dimensions as 'in_tensor' containing the softmax of 'in_tensor'
    """
    in_tensor = torch.exp(in_tensor)
    sum_ = torch.unsqueeze(torch.sum(in_tensor, 1), 1)
    return torch.div(in_tensor, sum_)


def softmaxAttack(model, data):
    """ Produces the scores for the Softmax attack, which is the maximum value of the softmax vector for each sample
    model: instance of a nn.Module subclass
    data: samples to be tested. Pytorch tensor with shape appropriate shape for 'model'
    outputs -> pytorch tensor of dimensions [batch] containing the softmax score of the input
    """
    scores, _ = torch.max(Softmax(model(data).detach()), 1)
    return scores

def advAttack(model, images, labels, batch_size=10, eps=3/255, alpha=0.001, steps=50, random_start=True, targeted=False):
    if cuda:
        dev = 'cuda'
    else:
        dev = 'cpu'
    
    iternum = pgd_attack(
        model=model,
        images = images,
        labels = labels,
        eps = eps,
        alpha = alpha,
        steps = steps,
        random_start = True,
        targeted = targeted
    )

    return iternum

def simpleBlackAttack(model, images, labels, batch_size=10, max_iters=300, freq_dims=32, stride=7, epsilon=0.05, linf_bound=0.0, order='rand', targeted=False, pixel_attack=False, log_every=40):
    if cuda:
        dev = 'cuda'
    else:
        dev = 'cpu'
    
    simpleblack_nums = simpleblackbox(
        model = model,
        images = images,
        labels = labels,
        max_iters = max_iters,
        freq_dims = freq_dims,
        stride = stride,
        epsilon = epsilon,
        linf_bound = linf_bound,
        order = order,
        targeted = targeted,
        pixel_attack =pixel_attack,
        log_every = log_every
    )
    return simpleblack_nums

def block_idct(x, block_size=8, masked=False, ratio=0.5, linf_bound=0.0):
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = np.zeros((x.size(0), x.size(1), block_size, block_size))
    if type(ratio) != float:
        for i in range(x.size(0)):
            mask[i, :, :int(block_size * ratio[i]), :int(block_size * ratio[i])] = 1
    else:
        mask[:, :, :int(block_size * ratio), :int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            submat = x[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)].cpu().numpy()
            if masked:
                submat = submat * mask
            z[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)] = torch.from_numpy(idct(idct(submat, axis=3, norm='ortho'), axis=2, norm='ortho'))
    if linf_bound > 0:
        return z.clamp(-linf_bound, linf_bound)
    else:
        return z

def labelonly_attack(model, images, labels, clip_max = 1, clip_min = 0, constraint = 'l2', num_iterations = 100, gamma = 1.0, stepsize_search = 'geometric_progression', max_num_evals = 1e4, init_num_evals = 100):
    if cuda:
        dev = 'cuda'
    else:
        dev = 'cpu'
    labelonly_iters = hsja_attack(
        model = model,
        images = images,
        labels = labels,
        clip_max = clip_max,
        clip_min = clip_min,
        constraint = constraint,
        num_iterations = num_iterations,
        gamma = gamma,
        stepsize_search = stepsize_search,
        max_num_evals = max_num_evals,
        init_num_evals = init_num_evals
    )
    return labelonly_iters

def un_labelonly_attack(model, images, labels, clip_max = 1, clip_min = 0, constraint = 'l2', num_iterations = 50, gamma = 1.0, stepsize_search = 'geometric_progression', max_num_evals = 1e4, init_num_evals = 100):
    if cuda:
        dev = 'cuda'
    else:
        dev = 'cpu'
    dists = un_hsja_attack(
        model = model,
        images = images,
        labels = labels,
        clip_max = clip_max,
        clip_min = clip_min,
        constraint = constraint,
        num_iterations = num_iterations,
        gamma = gamma,
        stepsize_search = stepsize_search,
        max_num_evals = max_num_evals,
        init_num_evals = init_num_evals
    )
    return dists
    
def square_attack_linf(model, images, labels, eps=0.05, n_iters=500, p_init=0.05, targeted=False, loss_type='margin-loss'):
    if cuda:
        dev = 'cuda'
    else:
        dev = 'cpu'
    square_iters = square_attack(
        model = model,
        images = images,
        labels = labels,
        eps = eps,
        n_iters = n_iters,
        p_init = p_init,
        targeted = targeted,
        loss_type = loss_type
    )
    return square_iters

def labelonly2017_attack(model, images, labels, epsilon=0.05, starting_delta_eps=0.0, max_queries=10000, sigma=1e-2, label_only_sigma=2e-3, zero_iters=100, max_lr=1e-2, min_lr=1e-3, momentum=0.9, plateau_length=20, plateau_drop=2.0, adv_thresh=0.2, conserative=2, samples_per_draw=100):
    if cuda:
        dev = 'cuda'
    else:
        dev = 'cpu'
    labelonly_iters = main_attack(
        model=model,
        images=images,
        labels=labels,
        epsilon=epsilon,
        starting_delta_eps=starting_delta_eps,
        max_queries=max_queries,
        sigma=sigma,
        label_only_sigma=label_only_sigma,
        zero_iters=zero_iters,
        max_lr=max_lr,
        min_lr=min_lr,
        momentum=momentum,
        plateau_length=plateau_length,
        plateau_drop=plateau_drop,
        adv_thresh=adv_thresh,
        conserative=conserative,
        samples_per_draw = samples_per_draw
    )
    return labelonly_iters

def advDistance(model, images, labels, batch_size=10, epsilon=1, norm='Linf'):
    """ Computes the adversarial distance score. First, adversarial examples are computed for each sample that is
    correctly classified by the target model. Then, the distance between the original and adversarial samples is
    computed. If a sample is misclassified, resulting adversarial distance will be 0.
    model: instance of a nn.Module subclass
    images: pytorch tensor with dimensions [batch,channels,width,height]
    labels: pytorch tensor of shape [batch] containing the integer labels of the 'images'
    batch_size: integer indicating the batch size for computing adversarial examples
    epsilon: maximum value for the magnitude of perturbations
    norm: indicates the norm used for computing adversarial examples and for measuring the distance between samples.
    Must be in {'Linf','L2','L1'}
    outputs -> pytorch tensor of dimensions [batch] containing the adversarial distance of 'images'
    """
    if norm == 'Linf':
        ordr = float('inf')
    elif norm == 'L1':
        ordr = 1
    elif norm == 'L2':
        ordr = 2

    if cuda:
        dev = 'cuda'
    else:
        dev = 'cpu'

    adversary = AutoAttack(model, norm=norm, eps=epsilon, version='standard', device=dev)
    adv = adversary.run_standard_evaluation(images, labels, bs=batch_size)

    dist = Dist(images, adv, ordr=ordr)

    return dist


def gradNorm(model, images, labels, Loss):
    """ Computes the l2 norm of the gradient of the loss w.r.t. the model parameters
    model: instance of a nn.Module subclass
    images: pytorch tensor with dimensions [batch,channels,width,height]
    labels: pytorch tensor of shape [batch] containing the integer labels of the samples
    loss: callable, loss function
    outputs -> pytorch tensor of dimensions [batch] containing the l2 norm of the gradients
    """
    loss = Loss(model(images), labels)
    gNorm = []
    for j in range(loss.shape[0]): # Loop over the batch
        grad_ = grad(loss[j], model.parameters(), create_graph=True)
        gNorm_ = -torch.sqrt(sum([grd.norm() ** 2 for grd in grad_]))
        gNorm.append(gNorm_.detach())
    return torch.stack(gNorm)


def lossAttack(model, images, labels, Loss):
    """ Computes the loss value for a batch of samples.
    model: instance of a nn.Module subclass
    images: pytorch tensor with dimensions [batch,channels,width,height]
    labels: pytorch tensor of shape [batch] containing the integer labels of the samples
    loss: callable, loss function
    outputs -> pytorch tensor of dimensions [batch] containing the negative loss values
    """
    loss = Loss(model(images).detach(), labels)
    return -loss


def Dist(sample, adv, ordr=float('inf')):
    """Computes the norm of the difference between two vectors. The operation is done for batches of vectors
    sample: pytorch tensor with dimensions [batch, others]
    adv: pytorch tensor with the same dimensions as 'sample'
    ordr: order of the norm. Must be in {1,2,float('inf')}
    outputs -> pytorch tensor of dimensions [batch] containing distance values for the batch of samples.
    """
    sus = sample - adv
    sus = sus.view(sus.shape[0], -1)
    return torch.norm(sus, ordr, 1)


def rescale01(data, Max, Min):
    """ Rescale input features to [0,1]. 
    data: pytorch tensor of shape [batch,features].
    Max: Tensor with shape [1].
    Min: Tensor with shape [1].
    """
    return (data - Min) / (Max - Min + 1e-8)


def Entropy(softprob):
    """ Compute the Shannon Entropy of a vector of soft probabilities.
    softprob: pytorch tensor. Vector of soft probabilities with shape [batch,num_classes]
    outputs -> pytorch tensor containing the entropy of each sample in the batch
    """
    epsilon = 1e-8
    return - torch.sum(softprob * torch.log(softprob + epsilon), 1)


def ModEntropy(softprob, labels):
    """Compute the modified entropy, described https://www.usenix.org/system/files/sec21fall-song.pdf, of a vector of
    soft probabilities.
    softprob: pytorch tensor. Vector of soft probabilities with shape [batch,num_classes]
    labels: pytorch tensor of shape [batch] containing the integer labels of the samples
    outputs -> pytorch tensor containing the modified entropy of each sample in the batch
    """
    epsilon = 1e-8
    confidence = torch.stack([softprob[i, labels[i]] for i in range(softprob.shape[0])])
    firstTerm = (confidence - 1) * torch.log(confidence + epsilon)
    secondTerm = - torch.sum(softprob * torch.log(1 - softprob + epsilon), 1)
    excessTerm = confidence * torch.log(1 - confidence + epsilon)
    return firstTerm + secondTerm + excessTerm
