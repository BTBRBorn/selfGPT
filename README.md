# About
This is my attempt to create an LLM. So far what I have gpt like model training
module on a single gpu. Because multi gpu training is expensive (quite expensive if you use thousands of them),
I am focusing on trying to find a much more efficient model architecture that has same computational properties
makes transformer architecture powerful.

But why implement everything from scratch. There are many great open source pretrained models I can finetune to
my needs, right? In my opinion, if you really want to understand every aspects of a system, you should build one from
scratch. And also I love doing it. It is so fun :). Another good reason, I want to invent much more efficient model
architecture than already existing ones (inference and/or training). We will still need multi gpu training of course but if I can invent a more efficient architecture, I can find investment much more easily to train a multi gpu model, and build a team to help me.

Since I wanted to experiment with different vocabulary sizes, I also wrote a tokenizer class that can be trained with BPE.

So far I used this module only with my laptop's gpu (Nvidia GeForce RTX 3600 6G). I got validation loss around 3.9 without even
trying (FineWebEdu 10BT dataset). This was only for confirming that everything works properly. I will run a real training run with a P100 and share the results.

# Usage
Even though module is written for personal use for now, I still think someone might want use it so I will share
code snippets here on how to train a model with it.
To download and process the data into shards in the form of numpy arrays, we use download_data.py. The dataset is FineWebEdu 10TB. It is hardcoded but if you like you can change it by changing one line inside download_data.py. You can just type following in terminal:

`python download_data.py`

Arguments:

    --shard_size_threshold: Threshold for shard sizes in terms of tokens.
    
    --data_path: Path of the folder to save all the shards.
    
    --tokenizer_path: If you trained and saved a tokenizer using tokenizer.py, by providing the path you can 
                      use it for tokenization.
    
    --streaming: Instead of downloading the dataset, you can set this parameter to 1. Then it will create the shards
                 without downloading the dataset.
    
    --tokens_threshold: Threshold for the total count of the tokens inside all shards. Default value is quite big so 
                        if don't provide a value whole datatset will be processed.
    

Examples:

`python download_data.py --tokenizer_path=path/to/tokenizer`

Datasets required for pretraining of LLMs are quite big so you may not want to download it in addition to the shards will be created in the data_path. In that case, you can provide --streaming and --tokens_threshold like this:
In this case, it will download the dataset and also sharding will stop at after you reach more 10 millions tokens.

`python download_data.py --streaming=1 --tokens_threshold=10000000`

After you download the data, you can start training with default arguments like this:

`python train.py`
Arguments:

    --learning_rate: Initial learning rate of the model.
    
    --max_iter: Number of training iterations.
    
    --num_batch_accum: Number of batches over which gradient will be accumulated.
    
    --batch_size: Batch size of the data.
    
    --n_layer: Number of Blocks inside the model.
    
    --vocab_size: Vocabulary size of the tokenizer.
    
    --block_size: Context length of the input.
    
    --n_head: Number of heads inside masked attention layer.
    
    --head_size: Output of each head
    
    --data_path: Path of the data
    
    --checkpoint_path: Path of the checkpoint. Model, optimizer and learning rate scheduler will be saved.

    --dataloader_num_workers: Number of processes will be used with dataloaders.
    
    --val_iter: Number of iterations will be used for evaluating the validation loss.
    
    --val_intervals: Validation loss will be checked every val_intervals.
    
    --resume_checkpoint: Instead of starting new, resume from checkpoint.
    
    --verbose: Whether or not print info about the training run.
    
    --compile_model: Whether or not use torch.compile to compile the model.
    

# Future Directions
So far I implemented pretraining part. For pretraining phase, one idea I had for a while to improve the transformer architecture is the replace every linear layer with something like Kolmogorov-Arnold Network. 

After pretraining, you can immediately do finetuning as well, by switching to a training set consists of Q&A
style text to get an AI agent.

Nowadays, post-training and inference time computing are quite important as well.
I am especially quite excited about inference time computing and self-play style reinforcement learning.
It seems to me that if we want to get more fluid intelligence, we need 
inference time training. And if we want these systems to be more capable than us at certain task, we also need self play
style reinforcement learning.
