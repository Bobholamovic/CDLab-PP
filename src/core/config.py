import sys
import argparse
import os.path as osp
from collections.abc import Mapping

import yaml


def _chain_maps(*maps):
    chained = dict()
    keys = set().union(*maps)
    for key in keys:
        vals = [m[key] for m in maps if key in m]
        if isinstance(vals[0], Mapping):
            chained[key] = _chain_maps(*vals)
        else:
            chained[key] = vals[0]
    return chained


def read_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    return cfg or {}


def parse_configs(cfg_path, inherit=True):
    # Read and parse config files
    if inherit:
        cfg_dir = osp.dirname(cfg_path)
        cfg_name = osp.basename(cfg_path)
        cfg_name, ext = osp.splitext(cfg_name)
        parts = cfg_name.split('_')
        cfg_path = osp.join(cfg_dir, parts[0])
        cfgs = []
        for part in parts[1:]:
            cfg_path = '_'.join([cfg_path, part])
            if osp.exists(cfg_path+ext):
                cfgs.append(read_config(cfg_path+ext))
        cfgs.reverse()
        if len(parts)>=2:
            return _chain_maps(*cfgs, dict(tag=parts[1], suffix='_'.join(parts[2:])))
        else:
            return _chain_maps(*cfgs)
    else:
        return read_config(cfg_path)


def parse_args(parser_configurator=None):
    # Check if a config file is specified
    cfg_parser = argparse.ArgumentParser(add_help=False)
    cfg_parser.add_argument('--exp_config', type=str, default='')
    cfg_parser.add_argument('--inherit_off', action='store_true')
    cfg_args = cfg_parser.parse_known_args()[0]
    cfg_path = cfg_args.exp_config
    inherit_on = not cfg_args.inherit_off

    # Main parser
    parser = argparse.ArgumentParser(conflict_handler='resolve', parents=[cfg_parser])
    # Global settings
    parser.add_argument('cmd', choices=['train', 'eval'])

    # Data
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--subset', type=str, default='val')

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--load_optim', action='store_true')
    parser.add_argument('--save_optim', action='store_true')
    parser.add_argument('--sched_on', action='store_true')
    parser.add_argument('--schedulers', type=dict, nargs='*')

    # Training related
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--anew', action='store_true',
                        help="clear history and start from epoch 0 with model weights updated")
    parser.add_argument('--device', type=str, default='cpu')

    # Experiment
    parser.add_argument('--exp_dir', default='../exp/')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--debug_on', action='store_true')
    parser.add_argument('--log_off', action='store_true')
    parser.add_argument('--track_intvl', type=int, default=1)

    # Criterion
    parser.add_argument('--criterion', type=str)

    # Model
    parser.add_argument('--model', type=str)

    if parser_configurator is not None:
        parser = parser_configurator(parser)
    
    if osp.exists(cfg_path):
        cfg = parse_configs(cfg_path, inherit_on)
        
        def _cfg2args(cfg, parser, prefix=''):
            for k, v in cfg.items():
                opt = prefix+k
                if isinstance(v, (list, tuple)):
                    # Only apply to homogeneous lists or tuples
                    parser.add_argument('--'+opt, type=type(v[0]), nargs='*', default=v)
                elif isinstance(v, dict):
                    # Recursively parse a dict
                    _cfg2args(v, parser, opt+'.')
                elif isinstance(v, bool):
                    parser.add_argument('--'+opt, action='store_true', default=v)
                else:
                    parser.add_argument('--'+opt, type=type(v), default=v)
            return parser

        parser = _cfg2args(cfg, parser, '')
        args = parser.parse_args()
    elif cfg_path != '':
        raise FileNotFoundError
    else:
        args = parser.parse_args()

    def _args2cfg(cfg, args):
        args = vars(args)
        for k, v in args.items():
            pos = k.find('.')
            if pos != -1:
                # Iteratively parse a dict
                dict_ = cfg
                while pos != -1:
                    dict_.setdefault(k[:pos], {})
                    dict_ = dict_[k[:pos]]
                    k = k[pos+1:]
                    pos = k.find('.')
                dict_[k] = v
            else:
                cfg[k] = v
        return cfg

    return _args2cfg(dict(), args)