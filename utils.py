import torch
from pathlib import Path
import model
from typing import Any

def save_checkpoint(checkpoint_path, model, optimizer, scheduler, config, results):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'config': config,
        'results': results 
    }, Path(checkpoint_path))

def load_checkpoint(checkpoint_path) -> dict[str, Any]:
    checkpoint = torch.load(Path(checkpoint_path), weights_only=False) 
    config = checkpoint['config']

    gpt = model.GPT(config).to(config.device)
    gpt.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.Adam(gpt.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scheduler.load_state_dict(checkpoint['scheduler'])

    results = checkpoint['results']
    
    return_dict = {
                   'model': gpt,
                   'optimizer': optimizer,
                   'scheduler': scheduler,
                   'config': config,
                   'results': results
                  }

    return return_dict

