import model
from dataclasses import dataclass
import torch
from pathlib import Path
import tiktoken
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

@dataclass
class Config:
    batch_size: int = 8
    n_layer: int = 4
    vocab_size: int = 50257 
    context_size: int = 128
    block_size: int = 4
    n_head: int = 4
    head_size: int = 8
    n_embd: int = n_head*head_size

config = Config()

gpt = model.GPT(config).to(device)

data_path = Path('input.txt')
with open(data_path, 'r') as f:
    text = f.read()

tokenizer = tiktoken.get_encoding('gpt2')
tokens = tokenizer.encode(text)
tokens = torch.tensor(tokens, dtype=torch.long)

B, T = config.batch_size, config.block_size
buff = tokens[:B*T+1]
x, y = buff[:B*T].view(B,T), buff[1:B*T+1].view(B,T)
x, y = x.to(device), y.to(device)
optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3)

gpt.train()
for _ in range(1000):
    B, T = x.size()
    optimizer.zero_grad()
    logits = gpt(x)
    loss = F.cross_entropy(logits.view(B*T, config.vocab_size), y.view(B*T))
    print(f'Loss: {loss.item()}')
    loss.backward()
    optimizer.step()

    