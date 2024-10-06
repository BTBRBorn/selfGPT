import torch
from pathlib import Path
import model
from typing import Any

def save_checkpoint(checkpoint_path, model, optimizer, config, results):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'results': results 
    }, Path(checkpoint_path))

def load_checkpoint(checkpoint_path, learning_rate) -> dict[str, Any]:
    checkpoint = torch.load(Path(checkpoint_path), weights_only=False) 
    config = checkpoint['config']

    gpt = model.GPT(config).to(config.device)
    gpt.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.Adam(gpt.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #load_state_dict overwrites learning rate at initialization
    #This is how we set it up to learning_rate pass down to the function
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    results = checkpoint['results']
    
    return_dict = {
                   'model': gpt,
                   'optimizer': optimizer,
                   'config': config,
                   'results': results
                  }

    return return_dict

