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

parser.add_argument('--learning_rate', type=float, default=6e-4)
parser.add_argument('--max_iter', type=int, default=500)
#We will use gradient accumulation
parser.add_argument('--num_batch_accum', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_layer', type=int, default=8)
#default tokenizer gpt2 has a vocab_size of 50257 but 50304 is divisible by 2,8,16,32 so it is
#better in terms of cuda optimization
parser.add_argument('--vocab_size', type=int, default=50304)
parser.add_argument('--block_size', type=int, default=256)
parser.add_argument('--n_head', type=int, default=12)
parser.add_argument('--head_size', type=int, default=32)
parser.add_argument('--data_path', type=str, default='data/')
parser.add_argument('--checkpoint_path', type=str, default='checkpoints/')
parser.add_argument('--dataloader_num_workers', type=int, default=4)
parser.add_argument('--val_iter', type=int, default=100)
parser.add_argument('--val_intervals', type=int, default=100)
parser.add_argument('--resume_checkpoint', type=int, default=0)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--compile_model', type=int, default=0)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Use tf32 precision for matrix multiplication
torch.set_float32_matmul_precision('high')

@dataclass
class Config:
    learning_rate: float = args.learning_rate
    max_iter: int = args.max_iter
    num_batch_accum: int = args.num_batch_accum
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
    val_intervals: int = args.val_intervals
    resume_checkpoint: bool = args.resume_checkpoint
    verbose: int = args.verbose
    compile_model: int = args.compile_model
    n_embd: int = n_head*head_size
    device: str = device

config = Config()

checkpoint_path = Path(config.checkpoint_path)
if not checkpoint_path.exists():
    checkpoint_path.mkdir()

#Whether or not resuming the training from a checkpoint
if not config.resume_checkpoint:
    gpt = model.GPT(config).to(device)
    optimizer = torch.optim.Adam(gpt.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), fused=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
    results = {'train_loss': [], 'val_loss': []}
else:
    return_dict = utils.load_checkpoint(checkpoint_path / Path('checkpoint.tar'))
    gpt = return_dict['model']
    optimizer = return_dict['optimizer']
    scheduler = return_dict['scheduler']
    results = return_dict['results']

if config.verbose:

    print('-'*50)
    print(f'Device: {device}')
    print('-'*50)

    print('-'*50)
    total_num_tokens = config.batch_size * config.block_size * config.num_batch_accum
    print(f'Total number of tokens processed in every training step: {total_num_tokens}')
    print('-'*50)


    print('-'*50)
    print('Model Description:')
    print(gpt)
    print('-'*50)

    print('-'*50)
    print('Total number of parameters:', sum([p.numel() for p in gpt.parameters()]))
    print('-'*50)

    print('-'*50)
    print('Optimizer Description:')
    print(optimizer)
    print('-'*50)

#compilation requires extra time so if you are only test/develop
#wouldn't recommend it. But when it is time to really train
#it could speed up the process about 2x
if config.compile_model:
    gpt = torch.compile(gpt)

#Get the meta_data from data/ folder
with open(Path(config.data_path) / Path('meta_data.pickle'), 'rb') as handle:
    meta_data = pickle.load(handle)

train_dataloader, val_dataloader = get_dataloader.create_dataloaders(meta_data, config)
train_iter = iter(train_dataloader)
val_iter = iter(val_dataloader)

gpt_results = engine.train(gpt, train_iter, val_iter, train_dataloader, val_dataloader,
                           optimizer, scheduler, config, results) 

utils.save_checkpoint(checkpoint_path / Path('checkpoint.tar'), gpt, optimizer, scheduler, config, gpt_results)

#Generate text with the model. By default it will generate 5 different versions.
tokenizer = tiktoken.get_encoding('gpt2')
gpt.generate('I know Kung Fu.', tokenizer)
