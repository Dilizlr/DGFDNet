import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
from eval import _eval
from models.DGFDNet import build_net
from torch.backends import cudnn
from train import _train


def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net()
    # print(model)
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    if device != 'cpu':
        model.to(device)
    if args.mode == 'train':
        _train(model, args, device)

    elif args.mode == 'test':
        _eval(model, args, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='DGFDNet', type=str)

    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    parser.add_argument('--data_dir', type=str, default='/Home/zlr/Code/Hazydataset/Dense_Haze/')

    # Train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=5000)
    parser.add_argument('--print_freq', type=int, default=5)
    parser.add_argument('--num_worker', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--valid_freq', type=int, default=10)
    parser.add_argument('--resume', type=str, default='')


    # Test
    parser.add_argument('--test_model', type=str, default='')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])

    args = parser.parse_args()
    # 获取当前时间戳
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式化为年月日_时分秒
    args.model_save_dir = os.path.join('results/', args.model_name, 'Dense/', current_time)
    args.result_dir = os.path.join('results/', args.model_name, 'test/', current_time)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    print(args)
    main(args)
