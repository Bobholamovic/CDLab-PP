#!/usr/bin/env python3

import logging
from re import L
import shutil
import random
import os.path as osp

import paddle
import numpy as np

import core
import impl.builders
import impl.trainers
from core.misc import R
from core.config import parse_args
    

def main():
    # Set random seed
    RNG_SEED = 114514
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    paddle.seed(RNG_SEED)

    # Disable the extra handler
    logging.root.handlers.clear()

    # Parse commandline arguments
    def parser_configurator(parser):
        parser.add_argument('--crop_size', type=int, default=256, metavar='P', 
                            help="patch size (default: %(default)s)")
        parser.add_argument('--vdl_on', action='store_true')
        parser.add_argument('--vdl_intvl', type=int, default=100)
        parser.add_argument('--suffix_off', action='store_true')
        parser.add_argument('--save_on', action='store_true')
        parser.add_argument('--out_dir', default='')
        parser.add_argument('--weights', type=float, nargs='+', default=None)
        parser.add_argument('--argmax_on', action='store_true')

        return parser
        
    args = parse_args(parser_configurator)

    paddle.set_device(args['device'])

    trainer = R['Trainer_switcher'](args)

    if trainer is not None:
        if args['exp_config']:
            # Make a copy of the config file
            cfg_path = osp.join(trainer.gpc.root, osp.basename(args['exp_config']))
            shutil.copy(args['exp_config'], cfg_path)
        try:
            trainer.run()
        except BaseException as e:
            import traceback
            trainer.logger.fatal(traceback.format_exc())
            if args['debug_on']:
                import sys
                import pdb
                pdb.post_mortem(sys.exc_info()[2])
            exit(1)
    else:
        raise RuntimeError("Cannot find an appropriate trainer.")


if __name__ == '__main__':
    main()