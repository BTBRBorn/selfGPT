import torch
from torch import nn
import math
import torch.nn.functional as F

class MaskedSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.e_proj = nn.Linear(config.n_embd, 3*config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        #Add a flag for residual stream initialization
        assert not hasattr(self.proj, 'RESIDUAL_INIT_FLAG')
        self.proj.RESIDUAL_INIT_FLAG = 1
        self.register_buffer('tril', torch.tril(torch.ones((config.block_size, config.block_size))))

    def forward(self, x:torch.Tensor, flash_attention=True):
        B, T, C = x.size()
        QKV = self.e_proj(x)#(B, T, 3C)
        Q, K, V = QKV.split(dim=-1, split_size=self.config.n_embd) #(B, T, C), (B, T, C), (B, T, C)

        Q = Q.view(B, T, self.config.n_head, self.config.head_size).transpose(1,2)#Q: (B, nh, T, hs)
        K = K.view(B, T, self.config.n_head, self.config.head_size).transpose(1,2)#Q: (B, nh, T, hs)
        V = V.view(B, T, self.config.n_head, self.config.head_size).transpose(1,2)#Q: (B, nh, T, hs)
        if not flash_attention:
            att = (Q @ K.transpose(-1,-2)) / (math.sqrt(self.config.head_size))# (B, nh, T, T)
            att = att.masked_fill(self.tril[:T, :T]==0, float('-inf'))# (B, nh, T, T)
            att = F.softmax(att, dim=-1)# (B, nh, T, T)
            y = att @ V #(B, nh, T, hs)

        #There is no reason not to use flash attention as far as I know
        #I keeping above code for experimentation
        else:
            y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        y = y.transpose(1,2).reshape(B, T, C)

        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(4*config.n_embd, config.n_embd)

        #Add a flag for residual stream initialization
        assert not hasattr(self.proj, 'RESIDUAL_INIT_FLAG')
        self.proj.RESIDUAL_INIT_FLAG = 1
    def forward(self, x):
        return self.proj(self.gelu(self.linear(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.msa = MaskedSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.msa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        #lm_head and tok_emb will share the weights
        #this could increase the performance slightly and reduce the parameter count
        self.tok_emb.weight = self.lm_head.weight
        self.register_buffer('pos_inx', torch.arange(config.block_size, dtype=torch.long))

        #Initialize the weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Linear):
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            std = in_features**-0.5
            if hasattr(module, 'RESIDUAL_INIT_FLAG'):
                std *= (2*self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1 / (self.config.n_embd**0.5))

    def forward(self, x:torch.Tensor):
        B, T = x.size()
        x = self.tok_emb(x) + self.pos_emb(self.pos_inx[:T]) # tok_emb:(B,T,C), pos_emb:(T,C) -> x:(B,T,C)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        return self.lm_head(x)

    def generate(self, text:str, tokenizer, num_sequence: int = 5, num_generate: int = 100):
        self.eval()
        with torch.inference_mode():
            tokens = torch.tensor(tokenizer.encode_ordinary(text), dtype=torch.long, device=self.config.device)
            tokens = tokens.repeat(num_sequence).view(num_sequence, -1)
            for _ in range(num_generate):
                logits = self(tokens)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                probs_topk = torch.topk(probs, k=num_sequence).indices[0]
                tokens = torch.cat([tokens, probs_topk.view(num_sequence, -1)], dim=-1)

            for i in range(len(tokens)):
                print('-'*50)
                print(tokenizer.decode(tokens[i].tolist()))
                print('-'*50)
