import os
import sys
import yaml
import argparse
import datetime
import logging
import dill as pickle

import torch 
import numpy as np
from torch.utils.data import random_split

from model.MLP import * 
from model.CNN import * 
from model.RNN import * 
from data.dataset import * 
from train import * 
from test import test_model

def parse_args(): 
    description = "you should add those parameter"                   # 步骤二
    parser = argparse.ArgumentParser(description=description)        # 这些参数都有默认值，当调用parser.print_help()或者运行程序时由于参数不正确(此时python解释器其实也是调用了pring_help()方法)时，

    parser.add_argument('--model', type=str, help = "Choose a model: \{MLP/CNN/RNN\}")                   # 步骤三，后面的help是我的描述
    parser.add_argument('--lr', type=float, default=1e-3)                   # 步骤三，后面的help是我的描述
    parser.add_argument('--train_epochs', type=int, default=10)                   # 步骤三，后面的help是我的描述
    parser.add_argument('--alpha', type=float, default=1.0)                   # 步骤三，后面的help是我的描述
    parser.add_argument('--batch_size', type=int, default=256)                   # 步骤三，后面的help是我的描述
    parser.add_argument('--save_dir', type=str, default="")                   # 步骤三，后面的help是我的描述
    parser.add_argument('--layer_num', type=int, default=1)                   # 步骤三，后面的help是我的描述
    parser.add_argument('--hidden_size', type=int, default=128)                   # 步骤三，后面的help是我的描述
    parser.add_argument('--data_path', type=str, default="MNIST_data/")                   # 步骤三，后面的help是我的描述
    parser.add_argument('--activation', type=str, default="R")                   # 步骤三，后面的help是我的描述
    args = parser.parse_args()                                       # 步骤四          
    return args

def parse_yaml(model): 
    f = open("config/default_" + str(model) + ".yml", "r" ) 
    y = yaml.load(f, Loader=yaml.FullLoader) 
    return y 

if __name__ == "__main__": 
    yml_cfg = parse_yaml("MLP")

    args = parse_args()
    
    

    for i, j in yml_cfg.items(): 
        if not hasattr(args, str(i)): 
            setattr(args, str(i), j)

    if not hasattr(args, "exp_name"):
        args.exp_name = args.save_dir + "_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists(os.path.join("exp", args.exp_name)):
        os.makedirs(os.path.join("exp", args.exp_name))

    lr = args.lr
    alpha = args.alpha
    epoch = args.train_epochs
    batch_size = args.batch_size
    data_path = args.data_path

    if args.model == 'MLP': 
        model = MLP(784, 10, args.layer_num, args.hidden_size, args.activation) 
    model.optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    
    model.loss_func = torch.nn.CrossEntropyLoss()
    model.metric_func = lambda y_p, y: np.sum(torch.argmax(y_p.detach(), axis=1).numpy() == y.detach().numpy())
    model.metric_name = "ACC"

    data_name = ['train', 'valid', 'test']

    if not os.path.exists(data_path) : 
        full_dataset = MNIST("train", transform=lambda x: x / 255)
        train_num = int(len(full_dataset) * alpha)
        train_dataset, valid_dataset = random_split(full_dataset, [train_num, len(full_dataset) - train_num])  
        test_dataset = MNIST("test", transform=lambda x: x / 255) 

        os.mkdir(data_path) 
        for i in data_name:
            pickle.dump(eval(i + "_dataset"), open(data_path + i , "wb"))

    dataset = {}
    for i in data_name: 
        dataset[i] = pickle.load(open(data_path + i, "rb"))

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m/%d %I:%M:%S %p")

    fh = logging.FileHandler(os.path.join("exp", args.exp_name, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)

    # 配置文件
    with open(os.path.join("exp", args.exp_name, "config.yml"), "w") as f:
        yaml.dump(args, f)

    train_dl = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(dataset['valid'], batch_size=batch_size, shuffle=False) 
    test_dl = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False) 
    if len(dataset['valid']) == 0 : valid_dl = test_dl

    _, _, _, Train_acc = train_model(model, args.valid_epochs, train_dl, valid_dl, 50, dir="exp/" + args.exp_name)     
    logger.info("Train ACC: " + str(Train_acc))
    
    # train_model(model, args.valid_epochs, train_dl, valid_dl, 50) 
    
    model.load_state_dict(torch.load("exp/" + args.exp_name + "/net.pth"))    
    Test_acc = test_model(model, test_dl) 
    logger.info("Test ACC: " + str(Test_acc))
