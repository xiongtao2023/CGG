from utils.argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument('--model', required=True, default='CGG', help='the prediction model')
parser.add_argument(
    '--dataset-dir', type=Path, required=True, default='datasets/', help='the dataset set directory'
)
parser.add_argument(
    '--embedding-dim', type=int, default=128, help='the dimensionality of embeddings'
)
parser.add_argument(
    '--feat-drop', type=float, default=0.2, help='the dropout ratio for input features'
)
parser.add_argument(
    '--num-layers',
    type=int,
    default=2,
    help='the number of HGNN layers in the KGE component',
)
parser.add_argument(
    '--num-neighbors',
    default='10',
    help='the number of neighbors to sample at each layer.'
    ' Give an integer if the number is the same for all layers.'
    ' Give a list of integers separated by commas if this number is different at different layers, e.g., 10,10,5'
)
parser.add_argument(
    '--model-args',
    type=str,
    default='{}',
    help="the extra arguments passed to the model's initializer."   
    ' Will be evaluated as a dictionary.',
)
parser.add_argument('--batch-size', type=int, default=128, help='the batch size')
parser.add_argument(
    '--epochs', type=int, default=30, help='the maximum number of training epochs'
)
parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
parser.add_argument('--beta1', type=float, default=0.01)
parser.add_argument('--beta2', type=float, default=0.01)
parser.add_argument('--num_heads', type=int, default=2, help='')
parser.add_argument('--num_blocks', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument(
    '--weight-decay',
    type=float,
    default=1e-4,
    help='the weight decay for the optimizer',
)
parser.add_argument(
    '--patience',
    type=int,
    default=5,
    help='stop training if the performance does not improve in this number of consecutive epochs',
)
parser.add_argument(
    '--Ks',
    default='10,20,30',
    help='the values of K in evaluation metrics, separated by commas'
)
parser.add_argument(
    '--ignore-list',
    default='bias,batch_norm,activation',
    help='the names of parameters excluded from being regularized',
)
parser.add_argument(
    '--log-level',
    choices=['debug', 'info', 'warning', 'error'],
    default='debug',
    help='the log level',
)
parser.add_argument(
    '--log-interval',
    type=int,
    default=1000,
    help='if log level is info or debug, print training information after every this number of iterations',
)
parser.add_argument(
    '--device', type=int, default=-1, help='the index of GPU device (-1 for CPU)'
)
parser.add_argument(
    '--num-workers',
    type=int,
    default=1,
    help='the number of processes for data loaders',
)
parser.add_argument(
    '--OTF',
    action='store_true',
    help='compute KG embeddings on the fly instead of precomputing them before inference to save memory',
)
args = parser.parse_args()
args.model_args = eval(args.model_args)
args.num_neighbors = [int(x) for x in args.num_neighbors.split(',')]
args.Ks = [int(K) for K in args.Ks.split(',')]
args.ignore_list = [x.strip() for x in args.ignore_list.split(',') if x.strip() != '']

import logging
import importlib

module = importlib.import_module(f'{args.model}')
config = module.config
for k, v in vars(args).items():
    config[k] = v
args = config

log_level = getattr(logging, args.log_level.upper(), None)
logging.basicConfig(format='%(message)s', level=log_level)
logging.debug(args)

import torch as th
import pandas as pd
from torch.utils.data import DataLoader
from utils.data.load import read_dataset, AugmentedDataset
from utils.train_runner import TrainRunner
from utils.data.load import (build_knowledge_graph)

args.device = (
    th.device('cpu') if args.device < 0 else th.device(f'cuda:{args.device}')
)
args.prepare_batch = args.prepare_batch_factory(args.device)

def main():
    logging.info(f'reading dataset {args.dataset_dir}...')
    df_train, df_valid, df_test, stats = read_dataset(args.dataset_dir)

    df_edges = pd.read_csv(args.dataset_dir / 'edges.txt', sep='\t')
    args.knowledge_graph = build_knowledge_graph(df_train, df_edges)

    args.num_categorys = stats.num_categorys
    args.num_items = stats.num_items
    args.max_len = stats.max_len

    model = args.Model(**args, **args.model_args)
    model = model.to(args.device)
    logging.debug(model)

    read_sid = args.Model.__name__ == 'DGRec'
    train_set = AugmentedDataset(df_train, read_sid)
    valid_set = AugmentedDataset(df_valid, read_sid)
    test_set = AugmentedDataset(df_test, read_sid)

    if 'CollateFn' in args:
        collate_fn = args.CollateFn(**args)
        collate_train = collate_fn.collate_train
        collate_test = collate_fn.collate_test
    else:
        collate_train = collate_test = args.collate_fn

    args.model = model

    if 'BatchSampler' in config:
        logging.debug('using batch sampler')
        batch_sampler = config.BatchSampler(
            train_set, batch_size=args.batch_size, drop_last=True, seed=0
        )
        train_loader = DataLoader(
            train_set,
            batch_sampler=batch_sampler,
            collate_fn=collate_train,
            num_workers=args.num_workers,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            collate_fn=collate_train,
            num_workers=args.num_workers,
            drop_last=True,
            shuffle=True,
        )

    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        collate_fn=collate_test,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        collate_fn=collate_test,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False,
    )

    runner = TrainRunner(train_loader, valid_loader, test_loader, **args)
    logging.info('start training')
    results = runner.train(args.epochs, log_interval=args.log_interval)


if __name__ == '__main__':
    main()