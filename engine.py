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
    logits = model(x)
    loss = F.cross_entropy(logits.view(config.batch_size*config.block_size, config.vocab_size),
                           y.view(config.batch_size*config.block_size))
    loss.backward()
    optimizer.step()
    return loss.item()

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
    for i in tqdm(range(config.max_iter)):
        start = time.time()
        train_loss = train_step(model, train_iter, train_dataloader, optimizer, config)
        torch.cuda.synchronize()
        end = time.time()
        num_tokens_per_sec = num_tokens/(end-start)
        if i % config.print_intervals == 0:
            val_loss = val_step(model, val_iter, val_dataloader, config)
            results['train_loss'].append(train_loss)
            results['val_loss'].append(val_loss)
            print_str = f"Iter: {i}, Train Loss: {results['train_loss'][-1]}, " + \
                        f"Val Loss: {results['val_loss'][-1]}, tokens/sec: {num_tokens_per_sec:.2f}"
            print(print_str)
    #If already not printed, print the last result and add them to results dict
    if i % config.print_intervals != 0:
        val_loss = val_step(model, val_iter, val_dataloader, config)
        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)
        print_str = f"Iter: {i}, Train Loss: {results['train_loss'][-1]}, " + \
                    f"Val Loss: {results['val_loss'][-1]}, tokens/sec: {num_tokens_per_sec:.2f}"
        print(print_str)

    return results
