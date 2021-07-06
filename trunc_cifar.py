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
parser.add_argument('--epochs', default=150, type=int, help='number of epochs to train neural networks', required=False)
parser.add_argument('--lr', type=float, default=1e-1, help='learning rate', required=False)
parser.add_argument('--momentum', type=float, default=0.0, help='momentum', required=False)
parser.add_argument('--weight_decay', type=float, default=0.0, help='momentum', required=False)
parser.add_argument('--step_lr', type=float, default=1e-1, help='number of steps to take before before decaying learning rate', required=False)
parser.add_argument('--step_lr_gamma', type=float, default=.9, help='step learning rate of decay', required=False)
parser.add_argument('--custom_lr_multiplier', help='custom learning rate multiplier', required=False)
parser.add_argument('--adam', type=bool, action='store_true', help='adam optimizer', required=False)
parser.add_argument('--trials', type=int, default=1, help='number of trials to perform experiment', required=False)
parser.add_argument('--out_dir', type=int, help='directory name to hold results for experiment', required=True)
parser.add_argument('--data_path', type=str, help='path to CIFAR dataset in filesystem', required=True)
parser.add_argument('--workers', default=8, help='number of workers to use', required=False)
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training CIFAR network', required=False)
parser.add_argument('--logit_ball', type=float, default=7.5, help='radius of logit ball for truncation set', required=False)
parser.add_argument('--should_save_ckpt', action='store_true', help='whether or not to save DNN checkpoints during training', required=False)
parser.add_argument('--save_ckpt_iters', type=int, help='whether or not to save DNN checkpoints during training', required=False)
parser.add_argument('--log_iters', type=int, help='how often to log training metrics', required=False)

# CONSTANTS
# noise distributions
gumbel = Gumbel(0, 1)
num_classes = 10
# store paths
BASE_CLASSIFIER = '/base_classifier'
LOGIT_BALL_CLASSIFIER = '/logit_ball'
STANDARD_CLASSIFIER = '/standard_classifier'
# truncated dataset names for saving datasets
TRUNC_TRAIN_DATASET = 'trunc_train_'
TRUNC_VAL_DATASET = 'trunc_val_'
TRUNC_TEST_DATASET = 'trunc_test_'
# learning rates to iterate over
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
            noise = noise.cuda()
       #     noised = logits + noise.cuda()
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



def train_(path, loaders, seed, ds): 
    """
    *INTERNAL FUNCTION* train resnet18 model on CIFAR-10 dataset.
    Args: 
        path (str) : store path 
        loaders (Iterable) : iterable containing the DataLoaders for training/validating model
        ds (delphi.utils.Dataset) : dataset
    Returns: 
        best trained model, store 
    """
    # logging store
    ch.manual_seed(seed)
    out_store = store.Store(path)
    exp_id = out_store.exp_id
    setup_store_with_metadata(args, out_store)
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=ds)
    # train classifier on truncated dataset 
    config.args = args
    model = train.train_model(args, model, loaders, store=out_store, parallel=args.parallel)
    out_store.close()

    # load in best classifier from training process
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=ds, resume_path=path + exp_id + '/checkpoint.pt.best')
    # reopen store for this experiment
    out_store = store.Store(path, exp_id)
    return model, out_store


# evaluate training mdoels on datasets
def eval_(model, out_store, unseen, test, trunc_train, train_one): 
    """
    *INTERNAL FUNCTION* evaluate model on datasets.
    Args: 
        model (delphi.AttackerModel) : evaluate CIFAR-10 classifier on the truncated and non-truncated datasets
        out_store (cox.store.Store) : store to save results in
    Returns: 
        returns nothing
    """
    # test the standard model against the various datasets
    train.eval_model(args, model, unseen, out_store, table='unseen')
    train.eval_model(args, model, test, out_store, table='test')
    train.eval_model(args, model, trunc_train, out_store, table='trunc_train')
    train.eval_model(args, model, train_one, out_store, table='train_base')
    out_store.close()


def main(args, learning_rates):   
    """
    Iterate over the learning rates for training the base classifier, 
    truncated classifier, and the standard classifier on truncated data.
    """  
    # load datasets
    ds = CIFAR(data_path=args.data_path)
    dataset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
        download=True, transform=transform_)
    train_one, train_two = ch.utils.data.random_split(dataset, [25000, 25000], generator=Generator().manual_seed(0))
    train_one_loader = DataLoader(train_one, batch_size=args.batch_size,
        shuffle=args.shuffle, num_workers=args.workers)
    train_two_loader = DataLoader(train_two, batch_size=args.batch_size,
        shuffle=args.shuffle, num_workers=args.workers)
    test_set = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False,
        download=True, transform=transform_)
    test_loader = DataLoader(test_set, batch_size=128,
        shuffle=args.shuffle, num_workers=args.workers)

    for i in range(args.trials):
        # iterate over learning rates
        for lr in learning_rates: 
            # set learning rate
            args.__setattr__('lr', lr)
            # seed for training neural networks
            seed = ch.randint(low=0, high=100, size=(1, 1))

            # train and evaluate TruncatedCE classifier
            base_classifier, out_store = train_(args.out_dir + BASE_CLASSIFIER, (train_one_loader, test_loader), seed, ds)

            # calibrate base classifier
            temp = calibrate(test_loader, base_classifier)
            # truncate dataset using the calibrated classifier
            phi = LogitBallComplement(args.logit_ball)
            x_trunc, x_unseen, y_trunc, y_unseen = truncate(train_two_loader, base_classifier, phi, temp, cuda=True)
            trunc_train_loader = DataLoader(TruncatedCIFAR(x_trunc, y_trunc, transform=None), num_workers=args.workers, shuffle=args.shuffle, batch_size=args.batch_size)
            unseen_loader = DataLoader(TruncatedCIFAR(x_unseen, y_unseen, transform=None), num_workers=args.workers, shuffle=args.shuffle, batch_size=args.batch_size)

            # evalute base classifier on truncated and non-truncated datasets
            eval_(base_classifier, out_store, unseen_loader, test_loader, trunc_train_loader, train_one_loader)

            # train and evaluate TruncatedCE classifier
            delphi_, out_store = train_(args.out_dir + LOGIT_BALL_CLASSIFIER, (trunc_train_loader, test_loader), seed, ds)
            eval_(delphi_, out_store, unseen_loader, test_loader, trunc_train_loader, train_one_loader)

            # train and evaluate standard classifier trained on truncated data
            standard_model, out_store = train_(args.out_dit + STANDARD_CLASSIFIER, (trunc_train_loader, test_loader), seed, ds)
            eval_(standard_model, out_store, unseen_loader, test_loader, trunc_train_loader, train_one_loader)  


if __name__ == '__main__': 

    args = parser.parse_args()
    args = Parameters(args.__dict__)

    # set other default hyperparameters
    args.__setattr__('parallel', False)
    args.__setattr__('validation_split', .8)
    args.__setattr__('num_samples', 1000)
    args.__setattr__('shuffle', True)
    args.__setattr__('device', 'cuda' if ch.cuda.is_available() else 'cpu')


    print(args)




    # main(args, LEARNING_RATES)





