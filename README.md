# About
This is my attempt to create an LLM like Claude or chatGPT. So far what I have gpt like model training
module on a single gpu. Because multi gpu training is expensive (quite expensive if you use thousands of them)
I am focusing on trying to find a much more efficient model architecture that has same computational properties
makes transformer architecture powerful.


# Future Work
So far I was implemented pre-training. You can also do finetuning as well by switching to a training set consists of Q&A
style text. But nowadays post-training and inference time computing are quite important as well. I am especially quite
excited about inference time computing and self play style reinforcement learning.