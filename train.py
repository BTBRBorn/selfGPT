import model
from dataclasses import dataclass
import torch
from pathlib import Path
import torch.nn.functional as F
import get_dataloader
import engine
import argparse
import pickle
import tiktoken
import utils

parser = argparse.ArgumentParser()

parser.add_argument('--learning_rate', type=float, default=3e-4)
parser.add_argument('--max_iter', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_layer', type=int, default=4)
parser.add_argument('--vocab_size', type=int, default=50257)
parser.add_argument('--block_size', type=int, default=128)
parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--head_size', type=int, default=8)
parser.add_argument('--data_path', type=str, default='data/')
parser.add_argument('--checkpoint_path', type=str, default='checkpoints/')
parser.add_argument('--dataloader_num_workers', type=int, default=4)
parser.add_argument('--val_iter', type=int, default=100)
parser.add_argument('--print_intervals', type=int, default=100)
parser.add_argument('--resume_checkpoint', type=bool, default=False)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('-'*50)
print(f'Device: {device}')
print('-'*50)

@dataclass
class Config:
    learning_rate: float = args.learning_rate
    max_iter: int = args.max_iter
    batch_size: int = args.batch_size
    n_layer: int = args.n_layer
    vocab_size: int = args.vocab_size 
    block_size: int = args.block_size
    n_head: int = args.n_head
    head_size: int = args.head_size
    data_path: str = args.data_path
    checkpoint_path: str = args.checkpoint_path
    dataloader_num_workers: int = args.dataloader_num_workers
    val_iter: int = args.val_iter
    print_intervals: int = args.print_intervals
    resume_checkpoint: bool = args.resume_checkpoint
    n_embd: int = n_head*head_size
    device: str = device

config = Config()

checkpoint_path = Path(config.checkpoint_path)
if not checkpoint_path.exists():
    checkpoint_path.mkdir()

if not config.resume_checkpoint:
    gpt = model.GPT(config).to(device)
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=config.learning_rate)
else:
    gpt, optimizer, config = utils.load_checkpoint(checkpoint_path / Path('checkpoint.tar'))

print('-'*50)
print('Model Description:')
print(gpt)
print('-'*50)

print('-'*50)
print('Total number of parameters:', sum([p.numel() for p in gpt.parameters()]))
print('-'*50)

#Get the meta_data from data/ folder
with open(Path(config.data_path) / Path('meta_data.pickle'), 'rb') as handle:
    meta_data = pickle.load(handle)

train_dataloader, val_dataloader = get_dataloader.create_dataloaders(meta_data, config)
train_iter = iter(train_dataloader)
val_iter = iter(val_dataloader)

gpt_results = engine.train(gpt, train_iter, val_iter, train_dataloader, val_dataloader, optimizer, config) 

utils.save_checkpoint(checkpoint_path / Path('checkpoint.tar'), gpt, optimizer, config)

tokenizer = tiktoken.get_encoding('gpt2')
gpt.generate('This is a Language Model:', tokenizer)
