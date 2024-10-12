"""
This module will download fineweb 10BT dataset from HuggingFace Hub and
it will shard it into numpy arrays
"""
import argparse
from dataclasses import dataclass
import tiktoken
import multiprocessing as mp
import tokenizer as Tokenizer
from pathlib import Path
from datasets import load_dataset
import numpy as np
import os
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard_size_threshold', type=int, default=int(1e7))
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--tokenizer_path', type=str, default=None)
    parser.add_argument('--tokenizer_type', type=str, default='base', choices=['base', 'regex'])
    parser.add_argument('--streaming', type=int, default=0, choices=[0, 1])
    parser.add_argument('--tokens_threshold', type=int, default=int(1e20)) 
    args = parser.parse_args()

    @dataclass
    class Config:
        shard_size_threshold: int = args.shard_size_threshold
        data_path: str = args.data_path
        tokenizer_path: str | None = args.tokenizer_path
        tokenizer_type: str = args.tokenizer_type
        streaming: int = args.streaming
        tokens_threshold: int = args.tokens_threshold

    config = Config()

    #Initialize the Tokenizer, If you have a tokenizer saved comes from tokenizer.py
    #you can use that too by providing --tokenizer_path and --tokenizer_type arguments
    #If you don't it will use tiktoken gpt2 tokenizer
    if config.tokenizer_path is not None:
        if args.tokenizer_type == 'base':
            #vocab size will be overwritten so doesn't matter what number it is
            tokenizer = Tokenizer.BaseTokenizer(vocab_size = 288)
            tokenizer.load(Path(args.tokenizer_path))
        elif args.tokenizer_type == 'regex':
            #vocab size will be overwritten so doesn't matter what number it is
            tokenizer = Tokenizer.RegexTokenizer(vocab_size = 288)
            tokenizer.load(Path(args.tokenizer_path))
    else:
        #By default gpt2 tiktoken tokenizer will be used
        tokenizer = tiktoken.get_encoding('gpt2')

    data_path = Path(args.data_path)
    if not data_path.exists():
        data_path.mkdir()

    if config.streaming:
        dataset = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=True)
    else:
        dataset = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train')

    EOT = tokenizer._special_tokens['<|endoftext|>'] 
    
    def tokenize(doc):
        tokens = [EOT]
        tokens.extend(tokenizer.encode_ordinary(doc['text']))
        tokens_np = np.array(tokens)
        assert (tokens_np >= 0).all() and (tokens_np < 2**16).all(), 'Data is not suitable for np.uint16'
        tokens_np = tokens_np.astype(np.uint16)
        return tokens_np

    num_cpu = max(1, os.cpu_count()//2)

    NUM_TOKENS = 0
    NUM_SHARD = 1
    SHARD_SIZE_THRESHOLD = config.shard_size_threshold
    tokens_list = []
    meta_data = []
    total_token_count = 0
    with mp.Pool(processes=num_cpu) as pool:
        for tokens in pool.imap(tokenize, dataset, chunksize=64):
            tokens_list.append(tokens)
            NUM_TOKENS += len(tokens)
            total_token_count += len(tokens)
            if SHARD_SIZE_THRESHOLD < NUM_TOKENS:
                chunk_path = data_path / Path(f'shard_{NUM_SHARD}.npy')
                shard = np.concatenate(tokens_list)
                np.save(chunk_path, shard)
                meta_data.append({'num_shard':NUM_SHARD, 'shard_size':NUM_TOKENS})
                NUM_SHARD += 1
                NUM_TOKENS = 0
                tokens_list = []
            if total_token_count >= config.tokens_threshold:
                break

    
    if len(tokens_list) != 0:
        chunk_path = data_path / Path(f'shard_{NUM_SHARD}.npy')
        shard = np.concat(tokens_list)
        meta_data.append({'num_shard':NUM_SHARD, 'shard_size':NUM_TOKENS})
        np.save(chunk_path, shard)

    meta_data_path = data_path / Path('meta_data.pickle')
    sorted(meta_data, key=lambda x: x['num_shard'])
    for d in meta_data:
        d['path'] = (Path(config.data_path) / Path(f'shard_{d["num_shard"]}.npy')).absolute()
    with open(meta_data_path, 'wb') as handle:
        pickle.dump(meta_data, handle)
