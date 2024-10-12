# About
This is my attempt to create an LLM like Claude or chatGPT. So far what I have gpt like model training
module on a single gpu. Because multi gpu training is expensive (quite expensive if you use thousands of them),
I am focusing on trying to find a much more efficient model architecture that has same computational properties
makes transformer architecture powerful. I am all about efficency. Even if in this case it is paramount to make
the model more efficient, I love making a system more efficient without losing any performance in general.

We will still need multi gpu training of course but if I can invent a more efficient architecture, I can find
investment much more easily to train a multi gpu model.

Since I wanted to experiment with different vocabulary sizes, I also wrote a tokenizer class that can be trained with BPE.

So far I used it only with my laptop's gpu (Nvidia GeForce RTX 3600 6G). I got validation loss around 3.9 without even
trying (FineWebEdu 10BT dataset). This was only for confirming that everything works properly. I will run a real training run with a P100 and share the results.

# Usage
Even though module is written for personal use for now, I still think someone might want use it so I will share
code snippets here on how to train a model with it.
To download and process the data we use download_data.py. You can just type following in terminal:

`python download_data.py`

Using this way by default, it will download, tokenize with tiktoken gpt2 tokenize and shard FineWebEdu 10BT dataset into
the data/ folder. If you trained your own tokenizer using tokenizer.py and saved it to dick then you can provide the path as follows:

`python download_data.py --tokenizer_path=path/to/tokenizer`

Datasets required for pretraining LLMs are quite big so you may not want to download it in addition to the shards will be created in the data_path. In that case, you can provide --streaming and --tokens_threshold like this:

`python download_data.py --streaming=1 --tokens_threshold=10000000`

In this case, it will download the dataset and also sharding will stop at after you reach more 10 millions tokens.
There are other CLI arguments you can use (like shard_size), you can check them in download_data.py.

After you download the data, you can start training with default arguments like this:

`python train.py`

# Future Directions
So far I implemented pretraining part. You can also do finetuning as well by switching to a training set consists of Q&A
style text of course to get an AI agent. However, nowadays post-training and inference time computing are quite important as well. I am especially quite excited about inference time computing and self-play style reinforcement learning.
It seems to me that if we want to get more fluid intelligence (especially if your model is relatively small), we need
inference time training. And if we want these systems to be more capable than us at certain task, we also need self play
style reinforcement learning.