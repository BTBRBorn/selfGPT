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
trying (FineWebEdu 10TB dataset). This was only for confirming that everything works properly. I will run a real training run with a P100 and share the results.

# Usage
Even though module is written for personal use for now, I still think someone might want use it so I will share
code snippets here on how to train a model with it.
To download and process the data we use download_data.py. By default  You can just type following in terminal:

`python download_data.py`



# Future Work
So far I implemented pretraining part. You can also do finetuning as well by switching to a training set consists of Q&A
style text of course to get an AI agent. However, nowadays post-training and inference time computing are quite important as well. I am especially quite excited about inference time computing and self-play style reinforcement learning.
It seems to me that if we want to get more fluid intelligence (especially if your model is relatively small), we need
inference time training. And if we want these systems to be more capable than us at certain task, we also need self play
style reinforcement learning.