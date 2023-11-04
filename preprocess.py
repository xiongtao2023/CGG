import sys
from utils import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset',
    choices=['purchase', 'tmall', 'yoochoose1_64'],
    default='view',
    required=True,
    help='the dataset name',
)
parser.add_argument(
    '--input-dir',
    type=Path,
    default='datasets',
    help='the directory containing the raw data files',
)
parser.add_argument(
    '--output-dir',
    type=Path,
    default='datasets',
    help='the directory to store the preprocessed dataset',
)
parser.add_argument(
    '--train-split', type=float, default=0.6, help='the ratio of the training set'
)
parser.add_argument(
    '--max-len', type=int, default=50, help='the maximum session length'
)
args = parser.parse_args()
print(args)

FILENAMES = {
    'purchase': ['customer_purchase_clicks.csv', 'customer_purchase_edges.csv'],
    'tmall': ['tmall_click_clicks.csv', 'tmall_click_edges.csv'],
    'yoochoose1_64': ['yoochoose1_64_clicks.csv', 'yoochoose1_64_edges.csv'],
}

filenames = FILENAMES[args.dataset]
for filename in filenames:
    if not (args.input_dir / filename).exists():
        print(f'File {filename} not found in {args.input_dir}', file=sys.stderr)
        sys.exit(1)
clicks = args.input_dir / filenames[0]
edges = args.input_dir / filenames[1]

import numpy as np
import pandas as pd
from utils.data.preprocess import preprocess, update_id

print('reading dataset...')
if args.dataset == 'purchase':

    df_clicks = pd.read_csv(
        clicks,
        sep=',',
        skiprows=1,
        header=None,
        names=['category', 'sessionId', 'itemId', 'timestamp'],
    )
    df_clicks['timestamp'] = pd.to_datetime(df_clicks.timestamp, unit='ms')
    df_loc = None
    df_edges = pd.read_csv(
        edges,
        sep=',',
        skiprows=1,
        header=None,
        names=['itemId', 'category'],
    )
elif args.dataset == 'tmall':

    df_clicks = pd.read_csv(
        clicks,
        sep=',',
        skiprows=1,
        header=None,
        names=['category', 'sessionId', 'itemId', 'timestamp'],
        usecols=[0, 1, 2, 3],
        dtype={0: np.int32, 1: np.int32, 2: np.int32, 3: str}
    )
    df_clicks['timestamp'] = pd.to_datetime(df_clicks.timestamp, unit='ms')
    df_loc = None
    df_edges = pd.read_csv(
        edges,
        sep=',',
        skiprows=1,
        header=None,
        names=['itemId', 'category'],
        usecols=[0, 1],
        dtype={0: np.int32, 1: np.int32}
    )
elif args.dataset == 'yoochoose1_64':

    df_clicks = pd.read_csv(
        clicks,
        sep=',',
        skiprows=1,
        header=None,
        names=['sessionId', 'timestamp', 'itemId', 'category'],
        usecols=[0, 1, 2, 3],
        dtype={0: np.int32, 1: str, 2: np.int32, 3: np.int32}
    )
    df_clicks['timestamp'] = pd.to_datetime(df_clicks.timestamp, format="%Y-%m-%dT%H:%M:%S.%fZ")
    df_loc = None
    df_edges = pd.read_csv(
        edges,
        sep=',',
        skiprows=1,
        header=None,
        names=['itemId', 'category'],
        usecols=[0, 1],
        dtype={0: np.int32, 1: np.int32}
    )
else:
    print(f'Unsupported dataset {args.dataset}', file=sys.stderr)
    sys.exit(1)

# df_edges = df_edges[df_edges.itemId != df_edges.category]
df_clicks = df_clicks.dropna()
print('converting IDs to integers...')

df_clicks, df_edges = update_id(
    df_clicks, df_edges, colnames=['category']
)
if df_loc is None:
    df_clicks, df_edges = update_id(
        df_clicks, df_edges, colnames=['itemId']
    )
else:
    df_clicks, df_loc = update_id(df_clicks, df_loc, colnames='itemId')
df_clicks = df_clicks.sort_values(['sessionId', 'timestamp'])
np.random.seed(123456)
preprocess(df_clicks, df_edges, df_loc, args)
