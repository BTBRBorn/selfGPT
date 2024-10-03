"""
- BaseTokenizer: Given an encoding it applies Byte Pair Encoding (BPE).

- RegexTokenizer: Subclass of BaseTokenizer. It applies BPE on subtexts comes from
  applying regex_pattern. By default it uses GPT4 regex pattern, you can specify GPT2 or
  supply your own regex pattern.

  At the moment, there is no special tokens handling with these classes but of course
  we will be using an '<|endoftext|>' token while training. I will add special tokens to
  the classes later on.
"""

import regex as re
from pathlib import Path
import pickle

#Got these patterns from Andrej Karpathy's minbpe library: https://github.com/karpathy/minbpe
#And he got them from tiktoken library: https://github.com/openai/tiktoken
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

def get_stats(tokens, stats):
    for t in zip(tokens, tokens[1:]):
        stats[t] = stats.get(t, 0 ) + 1
    return stats

def merge(tokens, pair, idx):
    new_tokens = []
    index = 0
    while index < len(tokens):
        if index < len(tokens)-1 and tokens[index] == pair[0] and tokens[index+1] == pair[1]:
            new_tokens.append(idx)
            index += 2
        else:
            new_tokens.append(tokens[index])
            index += 1
    return new_tokens
        
class BaseTokenizer:
    def __init__(self, vocab_size, encoding='utf-8'):
        assert encoding in ('utf-8', 'utf-16', 'utf-32'), "Encoding has to be 'utf-8', 'utf-16' or 'utf-32'"
        assert vocab_size > 256, 'Vocab size has to be bigger than 256'
        self.vocab_size = vocab_size
        self.encoding = encoding
        self.vocab = {}
        self.merges = {}
        self._special_tokens = {}

    #Setting special tokens will only affect decode method.
    #Right now Tokenizer classes' encode function doesn't treat special tokens differently 
    def set_special_tokens(self, special_tokens_dict:dict[int, str]):
        for idx, token in special_tokens_dict.items():
            self.vocab[idx] = token.encode(encoding=self.encoding)
            self._special_tokens[token] = idx
            self.vocab_size += 1

    def train(self, text, special_tokens:list[str] = ['<|endoftext|>'], verbose=False):
        tokens = text.encode(encoding=self.encoding)
        num_merges = self.vocab_size - 256
        self.vocab = {idx:bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            idx = 256 + i
            stats = {}
            stats = get_stats(tokens, stats)
            pair = max(stats, key=stats.get)
            tokens = merge(tokens, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f'Byte pair {pair} merged into {idx}')

        special_tokens_dict = {}
        for i, token in enumerate(special_tokens):
            special_tokens_dict[self.vocab_size + i] = token
        self.set_special_tokens(special_tokens_dict)
            
        return tokens
    #To match tiktoken library's API method has named encode_ordinary instead encode    
    def encode_ordinary(self, text):
        tokens = text.encode(encoding=self.encoding)
        while len(tokens) >= 2:
            stats = {}
            stats = get_stats(tokens, stats)
            pair = min(stats, key=lambda x: self.merges.get(x, float('inf')))
            if pair not in self.merges:
                break
            tokens = merge(tokens, pair, self.merges[pair])
        return tokens
    
    def decode(self, tokens):
        byte_text = b''.join([self.vocab[idx] for idx in tokens])
        return byte_text.decode(encoding=self.encoding, errors='replace')

    def save(self, save_path: str | Path):
        path = Path(save_path)
        with open(path, 'wb') as handle:
            pickle.dump(self, handle)

    def load(self, load_path: str | Path):
        print(f'Warning: {self.__class__.__name__}.load method overwrites all class attributes including vocab_size')
        path = Path(load_path)
        with open(path, 'rb') as handle:
            loaded_tok = pickle.load(handle)
        assert self.__class__ == loaded_tok.__class__, \
               f'Self ({self.__class__.__name__}) and loaded tokenizer ({loaded_tok.__class__.__name__}) are not the same class'

        for att_name in loaded_tok.__dict__:
            if hasattr(self, att_name):
                self.__dict__[att_name] = loaded_tok.__dict__[att_name]


class RegexTokenizer(BaseTokenizer):
    def __init__(self, vocab_size, encoding='utf-8', regex_pattern=GPT4_SPLIT_PATTERN):
        super().__init__(vocab_size, encoding)
        self.regex_pattern = regex_pattern
        self.compiled_pattern = re.compile(self.regex_pattern)

    def train(self, text:str, special_tokens:list[str] = ['<|endoftext|>'], verbose=False):
        text_chunks = re.findall(self.compiled_pattern, text)
        token_chunks = [list(chunk.encode(encoding=self.encoding)) for chunk in text_chunks]
        num_merges = self.vocab_size - 256
        self.vocab = {idx:bytes([idx]) for idx in range(256)}
        for num_merge in range(num_merges):
            stats = {}
            for chunk in token_chunks:
                stats = get_stats(chunk, stats)
            pair = max(stats, key=stats.get)
            idx = 256 + num_merge
            for i, chunk in enumerate(token_chunks):
                token_chunks[i] = merge(chunk, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f'Byte pair {pair} is merged into {idx}')

        tokens = []
        for chunk in token_chunks:
            tokens.extend(chunk)

        special_tokens_dict = {}
        for i, token in enumerate(special_tokens):
            special_tokens_dict[self.vocab_size + i] = token
        self.set_special_tokens(special_tokens_dict)
         
        return tokens
