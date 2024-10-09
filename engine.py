import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import time

def train_step(model,
               train_iter,
               train_dataloader,
               optimizer,
               config):

    try:
        x, y = next(train_iter)
    except StopIteration: 
        train_iter = iter(train_dataloader)
        x, y = next(train_iter)

    x, y = x.to(config.device), y.to(config.device)

    optimizer.zero_grad()
    with torch.autocast(device_type=config.device, dtype=torch.bfloat16): #Mixed Precision
        logits = model(x)
        loss = F.cross_entropy(logits.view(config.batch_size*config.block_size, config.vocab_size),
                            y.view(config.batch_size*config.block_size))
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item(), norm

def val_step(model, 
             val_iter,
             val_dataloader,
             config):

    try:
        x, y = next(val_iter)
    except StopIteration: 
        val_iter = iter(val_dataloader)
        x, y = next(val_iter)
    x, y = x.to(config.device), y.to(config.device)

    total_loss = 0
    model.eval()
    with torch.inference_mode():
        for _ in range(config.val_iter):
            with torch.autocast(device_type=config.device, dtype=torch.bfloat16):#Mixed Precision
                logits = model(x)
                loss = F.cross_entropy(logits.view(config.batch_size*config.block_size, config.vocab_size),
                                    y.view(config.batch_size*config.block_size))
            total_loss += loss.item()
    model.train()
    return total_loss / config.val_iter

def train(model,
          train_iter,
          val_iter,
          train_dataloader,
          val_dataloader,
          optimizer,
          config,
          results):

    num_tokens = config.batch_size * config.block_size
    total_tokens = 0
    total_seconds = 0
    for i in tqdm(range(config.max_iter)):
        start = time.time()
        train_loss, norm = train_step(model, train_iter, train_dataloader, optimizer, config)
        torch.cuda.synchronize()
        end = time.time()
        total_tokens += num_tokens
        total_seconds += end-start
        if i % config.print_intervals == 0:
            tokens_per_sec = total_tokens / total_seconds
            total_tokens, total_seconds = 0, 0
            val_loss = val_step(model, val_iter, val_dataloader, config)
            results['train_loss'].append(train_loss)
            results['val_loss'].append(val_loss)
            print_str = f"Iter: {i}, Train Loss: {results['train_loss'][-1]}, " + \
                        f"Val Loss: {results['val_loss'][-1]}, tokens/sec: {tokens_per_sec:.2f} " + \
                        f"Norm: {norm:.2f}"
            print(print_str)
    #If already not printed, print the last result and add them to results dict
    if i % config.print_intervals != 0:
        tokens_per_sec = total_tokens / total_seconds
        val_loss = val_step(model, val_iter, val_dataloader, config)
        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)
        print_str = f"Iter: {i}, Train Loss: {results['train_loss'][-1]}, " + \
                    f"Val Loss: {results['val_loss'][-1]}, tokens/sec: {tokens_per_sec:.2f} " + \
                    f"Norm: {norm:.2f}"
        print(print_str)

    return results
