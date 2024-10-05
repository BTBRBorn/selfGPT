import torch
from pathlib import Path
import model

def save_checkpoint(checkpoint_path, model, optimizer, config):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, Path(checkpoint_path))

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(Path(checkpoint_path), weights_only=False) 
    config = checkpoint['config']

    gpt = model.GPT(config).to(config.device)
    gpt.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.AdamW(gpt.parameters(), config.learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return gpt, optimizer, config

