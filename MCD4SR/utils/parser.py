import os
import yaml
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type = str, default='Amazon', help = 'dataset')
    parser.add_argument('--exp_name', type = str, default='default', help = 'experiment name')
    parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--save_dir_root', type = str, default='./result', help = 'test used ckpt path')
    parser.add_argument("--lr_encoder", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--mu", type=float, default=2.0)
    parser.add_argument("--moct_t", type=float, default=5.0)
    parser.add_argument('--dataset', type=str, choices=['beauty', 'electronics', 'toys', 'office', 'home', 'clothing'], help='')
    parser.add_argument("--w_icla", type=float, default=0.4)
    parser.add_argument("--w_simw", type=float, default=1.0)
    parser.add_argument("--w_balw", type=float, default=0.1)
    parser.add_argument("--w_moct", type=float, default=0.1)
    parser.add_argument('--test', type=bool, default=False, help='is test')
    args = parser.parse_args()
    return args

def setup(args):
    args.config = './configs/Amazon/{}.yaml'.format(args.dataset)
    args.experiment_path = os.path.join(f'{args.save_dir_root}/experiments', args.benchmark, args.exp_name)
    if args.test:
        args.ckpts = os.path.join(f'{args.save_dir_root}/experiments', args.benchmark, args.ckpts)
    config = get_config(args)
    merge_config(config, args)
    create_experiment_dir(args)
    save_experiment_config(args)

def get_config(args):
    print('Load config yaml from %s' % args.config)
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config

def merge_config(config, args):
    for k, v in config.items():
        setattr(args, k, v)   

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    
def save_experiment_config(args):
    config_path = os.path.join(args.experiment_path,'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(args.__dict__, f)
        print('Save the Config file at %s' % config_path)