"""
Script for running a batch job for truncated neural networks CIFAR-10 experiment. For this particular 
experiment, we use a truncation set that checks the difference between the two maximum logits from 
the Deep Neural Network.
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
from torch.utils.data import DataLoader, TensorDataset
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
import os
import config
import pickle
import pandas as pd
from argparse import ArgumentParser
import io
import json
import os
import pickle
import numpy as np
import scipy.stats
import pathlib
import PIL.Image

# set environment variable so that stores can create output files
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

parser = ArgumentParser(description='Parser for running Truncated CIFAR-10 Experiments')
parser.add_argument('--epochs', type=int, default=150,  help='number of epochs to train neural networks', required=False)
parser.add_argument('--lr', type=float, default=1e-1, help='learning rate', required=False)
parser.add_argument('--momentum', type=float, default=0.0, help='momentum', required=False)
parser.add_argument('--weight_decay', type=float, default=0.0, help='momentum', required=False)
parser.add_argument('--step_lr', type=float, default=1e-1, help='number of steps to take before before decaying learning rate', required=False)
parser.add_argument('--step_lr_gamma', type=float, default=.9, help='step learning rate of decay', required=False)
parser.add_argument('--custom_lr_multiplier', help='custom learning rate multiplier', required=False)
parser.add_argument('--adam', action='store_true', help='adam optimizer', required=False)
parser.add_argument('--trials', type=int, default=1, help='number of trials to perform experiment', required=False)
parser.add_argument('--out_dir', type=str, help='directory name to hold results for experiment', required=True)
parser.add_argument('--data_path', type=str, help='path to CIFAR dataset in filesystem', required=True)
parser.add_argument('--workers', type=int, default=8, help='number of workers to use', required=False)
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training CIFAR network', required=False)
parser.add_argument('--epsilon', type=float, default=.05, help='epsilon difference for the truncation set', required=False)
parser.add_argument('--should_save_ckpt', action='store_true', help='whether or not to save DNN checkpoints during training', required=False)
parser.add_argument('--save_ckpt_iters', type=int, help='whether or not to save DNN checkpoints during training', required=False)
parser.add_argument('--log_iters', type=int, help='how often to log training metrics', required=False)

# CONSTANTS
# noise distributions
gumbel = Gumbel(0, 1)
num_classes = 10
# store paths
BASE_CLASSIFIER = '/base_classifier/'
DIFF_TRUNC_SET = '/diff_trunc_set/'
STANDARD_CLASSIFIER = '/standard_classifier/'

# HELPER CODE
transform_ = transforms.Compose(
    [transforms.ToTensor()])

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


def train_(args, path, loaders, seed, ds, loss_fn, phi=None, cifar_10_1_loader=None): 
    """
    *INTERNAL FUNCTION* train resnet18 model on CIFAR-10 dataset.
    Args: 
        path (str) : store path 
        loaders (Iterable) : iterable containing the DataLoaders for training/validating model
        ds (delphi.utils.Dataset) : dataset
        loss_fn (ch.autograd.Function) : loss function for training neural network
        cifar_10_1 (ch.utils.data.DataLoader) : CIFAR-10.1 dataloader; if provided, track accuracy each epoch
    Returns: 
        best trained model, store 
    """
    # logging store
    ch.manual_seed(seed)
   
    # epoch hook logs CIFAR-10.1 accuracy
    if 'epoch_hook' in args: args.__delattr__('epoch_hook')
    out_store = store.Store(path)
    setup_store_with_metadata(args, out_store)
    epoch_hook = EpochHook(args, out_store, cifar_10_1_loader, loaders[1])
    args.__setattr__('epoch_hook', epoch_hook)

    # train model
    exp_id = out_store.exp_id
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=ds)
    # train classifier on truncated dataset 
    config.args = args
    model = train.train_model(args, model, loaders, store=out_store, parallel=args.parallel, criterion=loss_fn, phi=phi)
    out_store.close()

    # load in best classifier from training process
    resume_path = path + exp_id +  '/checkpoint.pt.best'
    # if there is not best network, take the most recent one
    if not os.path.exists(resume_path):
        resume_path = path + exp_id + '/checkpoint.pt.latest'

    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=ds, resume_path=resume_path)
    # reopen store for this experiment
    out_store = store.Store(path, exp_id)
    return model, out_store


def load_new_test_data(version_string='', load_tinyimage_indices=False):
    data_path = os.path.abspath('/home/gridsan/stefanou/data/CIFAR-10.1/datasets/')
    filename = 'cifar10.1'
    if version_string == '':
        version_string = 'v7'
    if version_string in ['v4', 'v6', 'v7']:
        filename += '_' + version_string
    else:
        raise ValueError('Unknown dataset version "{}".'.format(version_string))
    label_filename = filename + '_labels.npy'
    imagedata_filename = filename + '_data.npy'
    label_filepath = os.path.abspath(os.path.join(data_path, label_filename))
    imagedata_filepath = os.path.abspath(os.path.join(data_path, imagedata_filename))
    print('Loading labels from file {}'.format(label_filepath))
    assert pathlib.Path(label_filepath).is_file()
    labels = np.load(label_filepath)
    print('Loading image data from file {}'.format(imagedata_filepath))
    assert pathlib.Path(imagedata_filepath).is_file()
    imagedata = np.load(imagedata_filepath) / 255.0
    assert len(labels.shape) == 1
    assert len(imagedata.shape) == 4
    assert labels.shape[0] == imagedata.shape[0]
    assert imagedata.shape[1] == 32
    assert imagedata.shape[2] == 32
    assert imagedata.shape[3] == 3
    if version_string == 'v6' or version_string == 'v7':
        assert labels.shape[0] == 2000
    elif version_string == 'v4':
        assert labels.shape[0] == 2021

    if not load_tinyimage_indices:
        return imagedata, labels
    else:
        ti_indices_data_path = os.path.join(os.getcwd(), '/drive/MyDrive/Truncated Computer Vision/CIFAR-10.1/other_data/')
        ti_indices_filename = 'cifar10.1_' + version_string + '_ti_indices.json'
        ti_indices_filepath = os.path.abspath(os.path.join(ti_indices_data_path, ti_indices_filename))
        print('Loading Tiny Image indices from file {}'.format(ti_indices_filepath))
        assert pathlib.Path(ti_indices_filepath).is_file()
        with open(ti_indices_filepath, 'r') as f:
            tinyimage_indices = json.load(f)
        assert type(tinyimage_indices) is list
        assert len(tinyimage_indices) == labels.shape[0]
        return imagedata, labels, tinyimage_indices


class EpochHook: 
  def __init__(self, args, store, cifar_10_1, test_loader): 
    import copy 
    self.args = copy.deepcopy(args)
    self.store = store
    self.cifar_10_1_loader = cifar_10_1
    self.test_loader = test_loader


  def __call__(self, model, i):
    train.eval_model(self.args, model, self.test_loader, self.store, table='test')
    train.eval_model(self.args, model, self.cifar_10_1_loader, self.store, table='cifar-10-1')


class LogitDiffTruncSet: 
  """
  Truncation based off the difference between the largest and the second largest 
  logts. If the difference between the logits is larger than epsilon, then it falls 
  outside the truncation set.
  """
  def __init__(self, epsilon): 
    self.eps = epsilon

  def __call__(self, x): 
    topk = ch.topk(x, 2, dim=-1).values
    return (topk[...,0] - topk[...,1]) > self.eps


def main(args):   
    """
    Iterate over the learning rates for training the base classifier, 
    truncated classifier, and the standard classifier on truncated data.
    """  
    # load datasets
    ds = CIFAR(data_path=args.data_path)
    # training dataset
    dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=True,
        download=True, transform=transform_)
    train_loader = DataLoader(dataset, batch_size=args.batch_size,
        shuffle=args.shuffle, num_workers=args.workers)
    # testing dataset
    test_set = torchvision.datasets.CIFAR10(root=args.data_path, train=False,
        download=True, transform=transform_)
    test_loader = DataLoader(test_set, batch_size=128,
        shuffle=args.shuffle, num_workers=args.workers)

    # CIFAR-10.1 - Test Data
    cifar_10_1_test_data = load_new_test_data(version_string='v6')
    cifar_10_1_dataset = TensorDataset(Tensor(cifar_10_1_test_data[0]).permute(0, 3, 1, 2), Tensor(cifar_10_1_test_data[1]).long())
    cifar_10_1_loader = DataLoader(cifar_10_1_dataset, batch_size=128, num_workers=args.workers, shuffle=True)

    for i in range(args.trials):
        # seed for training neural networks
        seed = ch.randint(low=0, high=100, size=(1, 1))

        base_classifier, out_store = train_(args, args.out_dir + BASE_CLASSIFIER, (train_loader, test_loader), seed, ds, loss_fn=ch.nn.CrossEntropyLoss(), cifar_10_1_loader=cifar_10_1_loader)

        # set up difference in logits truncation set
        phi = LogitDiffTruncSet(args.epsilon)

        # evalute base classifier on truncated and non-truncated datasets

        # train and evaluate TruncatedCE classifier
        delphi_, out_store = train_(args, args.out_dir + DIFF_TRUNC_SET, (train_loader, test_loader), seed, ds, loss_fn=TruncatedCE.apply, phi=phi, cifar_10_1_loader=cifar_10_1_loader)


if __name__ == '__main__': 

    args = parser.parse_args()
    args = Parameters(args.__dict__)

    # set other default hyperparameters
    args.__setattr__('parallel', False)
    args.__setattr__('validation_split', .8)
    args.__setattr__('num_samples', 1000)
    args.__setattr__('shuffle', True)
    args.__setattr__('betas', (0.9, 0.999))
    args.__setattr__('amsgrad', False)
    args.__setattr__('accuracy', True)
    args.__setattr__('device', 'cuda' if ch.cuda.is_available() else 'cpu')
    print('args: ', args)

    # perform experiment
    main(args)




