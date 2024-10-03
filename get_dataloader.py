from torch.utils.data import Dataset
from pathlib import Path
import argparse
import os
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/')
args = parser.parse_args()

DATA_PATH = Path(args.data_path)

shard_list = os.listdir(DATA_PATH)

#print([s.split('_')[1].split('.')[0] for s in shard_list])

with open(DATA_PATH / Path('meta_data.pickle'), 'rb') as handle:
    meta_data = pickle.load(handle)

class CustomDataset(Dataset):
    def __init__(self, data_path_list: str | Path, ):
        self.paths = data_path_list
        