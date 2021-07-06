"""
Script for running a batch job for truncated neural networks CIFAR-10 experiment.
"""

from delphi import train
from delphi.utils import model_utils
from delphi import grad
from delphi import oracle
from delphi.utils.datasets import CIFAR
import delphi.utils.constants as consts
import delphi.utils.data_augmentation as da

import torchvision 
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch as ch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Gumbel
from torch import Generator
from tqdm.notebook import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
import cox
from cox.utils import Parameters
import cox.store as store
from cox.readers import CollectionReader
import seaborn as sns
import os
import config
import pickle
import pandas as pd
from argparse import ArgumentParser


# set environment variable so that stores can create output files
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

parser = ArgumentParser(description='Parser for running Truncated CIFAR-10 Experiments')
parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')

# CONSTANTS
# noise distributions
gumbel = Gumbel(0, 1)
num_classes = 10
# store paths
BASE_CLASSIFIER = '/home/gridsan/stefanou/cifar-10/resnet-18/base_classifier_noised_again'
LOGIT_BALL_CLASSIFIER = '/home/gridsan/stefanou/cifar-10/resnet-18/truncated_ce_classifier_noised_again'
STANDARD_CLASSIFIER = '/home/gridsan/stefanou/cifar-10/resnet-18/standard_classifier_noised_again'
# path to untruncated CV datasets
DATA_PATH = '/home/gridsan/stefanou/data/'
# truncated dataset names for saving datasets
TRUNC_TRAIN_DATASET = 'trunc_train_'
TRUNC_VAL_DATASET = 'trunc_val_'
TRUNC_TEST_DATASET = 'trunc_test_'

LEARNING_RATES = [1e-3, 1e-2, 1e-1, 2e-1, 3e-1]

# HELPER CODE
# dataset
class TruncatedCIFAR(Dataset):
    """
    Truncated CIFAR-10 dataset [Kri09]_.
    Original dataset has 50k training images and 10k testing images, with the
    following classes:
    * Airplane
    * Automobile
    * Bird
    * Cat
    * Deer
    * Dog
    * Frog
    * Horse
    * Ship
    * Truck
    .. [Kri09] Krizhevsky, A (2009). Learning Multiple Layers of Features
        from Tiny Images. Technical Report.
        
    Truncated dataset only includes images and labels from original dataset that fall within the truncation set.
    """
    def __init__(self, img, label, transform = None):
        """
        """
        self.img = img 
        self.label = label
        self.transform = transform

    def __getitem__(self, idx):
        """
        """
        x = self.img[idx]
        y = self.label[idx]
        # data augmentation
        if self.transform: 
            x = self.transform(x)
            
        return x, y
    
    def __len__(self): 
        return self.img.size(0)

transform_ = transforms.Compose(
    [transforms.ToTensor()])


# code for calibrating neural network
def T_scaling(logits, temp):
    """
    Temperature scaling.
    """
    return ch.div(logits, temp)


def calibrate(test_loader, model): 
    """
    Run SGD procedure to find temperature 
    scaling parameter.
    Args: 
        test_loader (ch.nn.DataLoader) : pytorch DataLoader with test dataset
        base_classifier (AttackerModel) : AttackerModel to calibrate
    Returns: ch.Tensor with the calculated temperature scalar
    """
    model.eval()
    temperature = ch.nn.Parameter(ch.ones(1).cuda())
    ce_loss = ch.nn.CrossEntropyLoss()

    # Removing strong_wolfe line search results in jump after 50 epochs
    optimizer = ch.optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')

    logits_list = []
    labels_list = []
    temps = []
    losses = []

    for images, labels in tqdm(test_loader, 0):
        images, labels = images.cuda(), labels.cuda()

        with ch.no_grad():
            logits_list.append(model(images)[0])
            labels_list.append(labels)

    # create tensors
    logits_list = ch.cat(logits_list).cuda()
    labels_list = ch.cat(labels_list).cuda()

    def _eval():
        loss = ce_loss(T_scaling(logits_list, temperature), labels_list)
        loss.backward()
        temps.append(temperature.item())
        losses.append(loss)
        return loss

    # run SGD - FUNCTIONAL PROGRAMMING
    optimizer.step(_eval)
    
    return temperature


# oracle
class LogitBallComplement: 
    """
    Truncation based off of complement norm of logits. Logit norm needs to be greater than input bound.
    In other words, retain the inputs that the classifier is more certain on. Larger 
    unnormalized log probabilities implies more certraining in classification.
    """
    def __init__(self, bound): 
        self.bound = bound

    def __call__(self, x): 
        return (x.norm(dim=-1) >= self.bound)

    def __str__(self): 
        return 'logit ball complement'


def truncate(loader, model, phi, temp, cuda=False): 
      """
      Truncate image dataset. 
      Args: 
        loader (ch.nn.DataLoader) : dataset to truncate
        model (delphi.AttackerModel) : model to truncate off of
        phi (delphi.oracle) : truncation set
        temp (ch.Tensor) : temperature scaling parameter to calibrate DNN by
        cuda (bool) : place dataset on GPU
      Returns: 
        Tuple with (x_trunc, x_trunc_test, y_trunc, y_trunc_test), where `_trunc` suffix
        indicates that the sampl feel within the truncation set, and the `_trunc_test` suffix 
        indicates that it didn't.
      """
      x_trunc, y_trunc = Tensor([]), Tensor([])
      # unseen test data
      x_trunc_test, y_trunc_test = Tensor([]), Tensor([])
      for inp, targ in loader: 
          # check if cuda
          if cuda: 
            inp, targ = inp.cuda(), targ.cuda()
          # pass images through base classifier
          logits, inp = model(inp) 
          # scaling logits by scaling parameter
          logits /= temp.item()
          noise = Gumbel(0, 1).rsample(logits.size())
          if cuda: 
       #     noise = noise.cuda()
            noised = logits + noise.cuda()
          noised = logits
          # truncate 
          filtered = phi(logits)
          indices = filtered.nonzero(as_tuple=False).flatten()
          test_indices = (~filtered).nonzero(as_tuple=False).flatten()
          x_trunc, y_trunc = ch.cat([x_trunc, inp[indices].cpu()]), ch.cat([y_trunc, targ[indices].cpu()])
          x_trunc_test, y_trunc_test = ch.cat([x_trunc_test, inp[test_indices].cpu()]), ch.cat([y_trunc_test, targ[test_indices].cpu()])
      return x_trunc, x_trunc_test, y_trunc.long(), y_trunc_test.long()
      

class TruncatedCE(ch.autograd.Function):
    """
    Truncated cross entropy loss.
    """
    @staticmethod
    def forward(ctx, pred, targ, phi):
        ctx.save_for_backward(pred, targ)
        ctx.phi = phi
        ce_loss = ch.nn.CrossEntropyLoss()
        return ce_loss(pred, targ)

    @staticmethod
    def backward(ctx, grad_output):  
        pred, targ = ctx.saved_tensors
        # initialize gumbel distribution
        gumbel = Gumbel(0, 1)
        # make num_samples copies of pred logits
        stacked = pred[None, ...].repeat(config.args.num_samples, 1, 1)
        # add gumbel noise to logits
        rand_noise = gumbel.sample(stacked.size()).to(config.args.device)
        noised = (stacked) + rand_noise 
        # truncate - if one of the noisy logits does not fall within the truncation set, remove it
        filtered = ctx.phi(noised)[..., None].to(config.args.device)
        noised_labs = noised.argmax(-1)
        # mask takes care of invalid logits and truncation set
        mask = noised_labs.eq(targ)[..., None] * filtered
        inner_exp = (1 - ch.exp(-rand_noise))
        avg = ((inner_exp * mask).sum(0) / (mask.sum(0) + 1e-5)) - ((inner_exp * filtered).sum(0) / (filtered.sum(0) + 1e-5))       
        return -avg / pred.size(0), None, None


def setup_store_with_metadata(args, store):
    '''
    Sets up a store for training according to the arguments object. See the
    argparse object above for options.
    '''
    args_dict = args.as_dict()
    schema = cox.store.schema_from_dict(args_dict)
    store.add_table('metadata', schema)
    store['metadata'].append_row(args_dict)


def main(args, learning_rates):   
    """
    Iterate over the learning rates for training the base classifier, 
    truncated classifier, and the standard classifier on truncated data.
    """  

    # load datasets
    ds = CIFAR(data_path=DATA_PATH)
    dataset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
        download=True, transform=transform_)
    train_one, train_two = ch.utils.data.random_split(dataset, [25000, 25000], generator=Generator().manual_seed(0))
    train_one_loader = ch.utils.data.DataLoader(train_one, batch_size=args.batch_size,
        shuffle=args.shuffle, num_workers=args.workers)
    train_two_loader = ch.utils.data.DataLoader(train_two, batch_size=args.batch_size,
        shuffle=args.shuffle, num_workers=args.workers)

    test_set = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False,
        download=True, transform=transform_)
    test_loader = ch.utils.data.DataLoader(test_set, batch_size=128,
        shuffle=args.shuffle, num_workers=args.workers)

    # iterate over learning rates
    for lr in learning_rates: 
        args.__setattr__('lr', lr)

        # train the base classifier
        ch.manual_seed(0)
        base_classifier, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=ds)
        out_store = store.Store(BASE_CLASSIFIER)
        setup_store_with_metadata(args, out_store)
        train.train_model(args, base_classifier, (train_one_loader, test_loader), store=out_store)

        # calibrate base classifier
        temp = calibrate(test_loader, base_classifier)

        # truncate dataset using the calibrated classifier
        phi = LogitBallComplement(args.logit_ball)
        x_trunc, x_trunc_test, y_trunc, y_trunc_test = truncate(train_two_loader, base_classifier, phi, temp, cuda=True)
        trunc_train_loader = DataLoader(TruncatedCIFAR(x_trunc, y_trunc, transform= None), num_workers=args.workers, shuffle=args.shuffle, batch_size=args.batch_size)
        trunc_test_loader = DataLoader(TruncatedCIFAR(x_trunc_test, y_trunc_test, transform= None), num_workers=args.workers, shuffle=args.shuffle, batch_size=args.batch_size)

        # test base classifier on datasets
        base_unseen_results = train.eval_model(args, base_classifier, trunc_test_loader, store=out_store, table='unseen')
        base_test_results = train.eval_model(args, base_classifier, test_loader, store=out_store, table='test')
        base_train_results = train.eval_model(args, base_classifier, trunc_train_loader, store=out_store, table='trunc_train')
        base_train_one_results = train.eval_model(args, base_classifier, train_one_loader, store=out_store, table='train_base')
        out_store.close()

        # logging store
        ch.manual_seed(0)
        delphi_, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=ds)
        out_store = store.Store(LOGIT_BALL_CLASSIFIER)
        setup_store_with_metadata(args, out_store)

        # train
        config.args = args
        delphi_ = train.train_model(args, delphi_, (trunc_train_loader, trunc_test_loader), store=out_store, phi=phi, criterion=TruncatedCE.apply)

        # test the standard model against the various datasets
        delphi_unseen_results = train.eval_model(args, delphi_, trunc_test_loader, out_store, table='unseen')
        delphi_test_results = train.eval_model(args, delphi_, test_loader, out_store, table='test')
        delphi_train_results = train.eval_model(args, delphi_, trunc_train_loader, out_store, table='trunc_train')
        delphi_train_one_results = train.eval_model(args, delphi_, train_one_loader, out_store, table='train_base')
        out_store.close()

        # logging store
        ch.manual_seed(0)
        out_store = store.Store(STANDARD_CLASSIFIER)
        setup_store_with_metadata(args, out_store)
        standard_model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=ds)

        # train classifier on truncated dataset 
        config.args = args
        standard_model = train.train_model(args, standard_model, (trunc_train_loader, trunc_test_loader), store=out_store, parallel=args.parallel)

        # test the standard model against the various datasets
        standard_unseen_results = train.eval_model(args, standard_model, trunc_test_loader, out_store, table='unseen')
        standard_test_results = train.eval_model(args, standard_model, test_loader, out_store, table='test')
        standard_train_results = train.eval_model(args, standard_model, trunc_train_loader, out_store, table='trunc_train')
        standard_train_one_results = train.eval_model(args, standard_model, train_one_loader, out_store, table='train_base')
        out_store.close()


if __name__ == '__main__': 
    # hyperparameters
    args = Parameters({ 
        'epochs': 100,
        'workers': 8, 
        'batch_size': 128, 
        'accuracy': True,
        'momentum': 0.9, 
        'weight_decay': 5e-4, 
        'save_ckpt_iters': 25,
        'step_lr': 10, 
        'step_lr_gamma': .9,
        'should_save_ckpt': True,
        'log_iters': 1,
        'validation_split': .8,
        'shuffle': True,
        'parallel': False, 
        'num_samples': 1000,
        'logit_ball': 7.5,
        'device': 'cuda' if ch.cuda.is_available() else 'cpu',
    })

    main(args, LEARNING_RATES)






