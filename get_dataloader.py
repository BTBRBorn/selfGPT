import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, meta_data:list[dict[str, Any]], config):
        #meta_data is the form: [{'num_shard':1, 'shard_size':1000000, 'path':Path object}] 
        self.meta_data = sorted(meta_data, key=lambda x: x['num_shard'])
        self.config = config

        num_examples = 0
        block_size = config.block_size
        for s_info in self.meta_data:
            num_examples += s_info['shard_size'] // (block_size + 1)
        self.num_examples = num_examples

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index: int) -> Any:
        num_examples = 0
        block_size = self.config.block_size
        for i, s_info in enumerate(self.meta_data):
            shard_path = s_info['path']
            shard_num_examples = s_info['shard_size'] // block_size 
            num_examples += shard_num_examples
            remainder = (index + 1) - num_examples
            if remainder <= 0:
                #In the very unlikely event that you don't have that one extra token for buff in the shard
                #code will choose the next shard's first example as x and y
                if remainder == 0 and (shard_num_examples*block_size == s_info['shard_size']):
                    shard_index = 0
                    tokens_np = np.load(self.meta_data[i+1]['path'])
                else:
                    shard_index = shard_num_examples + remainder - 1
                    tokens_np = np.load(shard_path)
                buff = tokens_np[shard_index*block_size:(shard_index+1)*block_size + 1]
                x, y = buff[:-1], buff[1:]
                x = torch.tensor(x, dtype=torch.long)
                y = torch.tensor(y, dtype=torch.long)
                return x, y
        raise IndexError(f'Index {index} out of range.')

def create_dataloaders(meta_data: dict[str, Any], config):
    
    num_workers = config.dataloader_num_workers
    meta_data_val = [min(meta_data, key=lambda x: x['shard_size'])]
    meta_data_train = [m for m in meta_data if id(m) != id(meta_data_val)]

    train_dataset = CustomDataset(meta_data_train, config)
    val_dataset = CustomDataset(meta_data_val, config)

    train_dataloaders = DataLoader(train_dataset, config.batch_size, shuffle=True,
                                   num_workers=num_workers, drop_last=True)
    val_dataloaders = DataLoader(val_dataset, config.batch_size, shuffle=True, drop_last=True)

    return train_dataloaders, val_dataloaders
