import argparse
import sys

from utils.config import options, update_options, reset_options
# from scheduler.trainer import Trainer
from scheduler import get_trainer
from utils.logger import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Training Entrypoint')
    parser.add_argument('--options', help='experiment options file name', required=False, type=str)

    args, rest = parser.parse_known_args()
    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)

    # training
    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument('--checkpoint', help='checkpoint file', type=str)
    parser.add_argument('--num-epochs', help='number of epochs', type=int)
    parser.add_argument('--version', help='version of task (timestamp by default)', type=str)
    parser.add_argument('--name', help='model name', type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger, writer = reset_options(options, args)
    set_random_seed(options.seed)
    trainer = get_trainer(options, logger, writer)
    trainer.train()


if __name__ == "__main__":
    main()
