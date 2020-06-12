import argparse
import time
from datetime import datetime
import os
from os.path import exists
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from data_loader import PolyphonicDataset
from data_loader import HumanMotionDataset
import models, configs
from helper import get_logger, gVar
from tensorboardX import SummaryWriter # install tensorboardX (pip install tensorboardX) before importing this package

def save_model(model, epoch):
    ckpt_path='./output/{}/{}/{}/models/model_epo{}.pkl'.format(args.model, args.expname, args.dataset, epoch)
    print("saving model to %s..." % ckpt_path)
    torch.save(model.state_dict(), ckpt_path)

def load_model(model, epoch):
    ckpt_path='./output/{}/{}/{}/models/model_epo{}.pkl'.format(args.model, args.expname, args.dataset, epoch)
    assert exists(ckpt_path), "epoch misspecified"
    print("loading model from %s..." % ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))

def random_select_data(test_set):
    N = test_set.__len__()
    index = int(np.random.uniform(0, N))
    return test_set.__getitem__(index)
    
# TODO
# write new function for model
'''
def prediction(model, seq, rev_seq, seq_len):
    model.generate(seq, rev_seq, seq_len)
'''

def main(args):
    # setup logging
    config=getattr(configs, 'config_'+args.model)()

    # instantiate the dmm
    model = getattr(models, args.model)(config)
    if args.reload_from>=0:
        load_model(model, args.reload_from)
    if args.data_path == './data/human_motion/':
    	train_set=HumanMotionDataset(args.data_path+'train.pkl')
    	valid_set=HumanMotionDataset(args.data_path+'valid.pkl')
    	test_set=HumanMotionDataset(args.data_path+'test.pkl')
    else:
    	train_set=PolyphonicDataset(args.data_path+'train.pkl')
    	valid_set=PolyphonicDataset(args.data_path+'valid.pkl')
    	test_set=PolyphonicDataset(args.data_path+'test.pkl')

    seq, rev_seq, seq_len = random_select_data(test_set)
    load_model(model, 449)
    prediction(model, seq, rev_seq, seq_len)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--data-path', type=str, default='./data/human_motion/')
    parser.add_argument('--model', type=str, default='DHMM', help='model name')
    parser.add_argument('--dataset', type=str, default='JSBChorales', help='name of dataset. SWDA or DailyDial')
    parser.add_argument('--expname', type=str, default='basic',
                        help='experiment name, for disinguishing different parameter settings')
    parser.add_argument('--reload_from', type=int, default=-1, help='reload from a trained ephoch')
    parser.add_argument('--test-freq', type=int, default = 50, help = 'frequency of evaluation in the test set')
    parser.add_argument('-v', '--visual', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('-l', '--log', type=str, default='dmm.log')
    args = parser.parse_args()

    os.makedirs(f'./output/{args.model}/{args.expname}/{args.dataset}/models', exist_ok=True)
    main(args)
